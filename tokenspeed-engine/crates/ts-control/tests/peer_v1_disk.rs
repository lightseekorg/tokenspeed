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

//! End-to-end multi-rank tests for v1 cross-layout Disk-tier peer prefix
//! sharing (`TP8/KVP8 -> TP8/KVP8` and `TP1/KVP1 -> TP8/KVP8`), including
//! hybrid `full_history` + `state` models, mixed Device+Disk landings,
//! byte-for-byte verification across all 8 reader ranks, and clean-abort vs
//! unknown-write-retirement fault isolation.

use std::{
    collections::HashMap,
    ffi::CString,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Once,
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use pyo3::{prelude::*, types::PyDict};
use ts_control::{
    disk::{DiskSettings, DiskTier},
    peer::{
        geometry::{DeviceField, HostGroup, HostPool, RankGeometry, Spans},
        landing::Timeouts,
        push::{
            PagePusher, PageWriter, PushEndpoint, PushItem, PushOutcome, PushSource, PushTarget,
        },
        read::Arena,
        registered::{Member, Replica as RegisteredReplica},
        Peer, PeerListener, PeerProgress, PeerSettings, PeerTransfers,
    },
};
use ts_disk::{Backend, ExecutorConfig, Memory};
use ts_l3::{GroupLayout, HostBuffer, HostLayout};
use ts_scheduler::{
    CacheEvent, CacheEventKind, CacheOpKind, DiskOpKind, DiskOpView, ForwardEvent,
    ForwardEventKind, PeerEventKind, PeerFetchView, PeerPageView, PeerTier, RequestSpec, Scheduler,
};

const PAGE: i32 = 4;
const BLOCK_BYTES: usize = 4096;
const STATE_GLOBAL_HEADS: usize = 8;
const STATE_HEAD_BYTES: usize = 512; // 8 * 512 == 4096 == BLOCK_BYTES
const HOLDER_DEVICE_LCM: usize = 5;
const READER_DEVICE_LCM: usize = 33;
const HOST_LCM_BLOCKS: usize = 8;
const DISK_LCM_BLOCKS: usize = 64;
const ARENA_BASE_OFFSET: u64 = 0x10_000; // Non-zero so registered regions have a > 0.

static NEXT_FILE_ID: AtomicU64 = AtomicU64::new(1);

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

fn hybrid_contract_wire(width: usize, device_lcm_pages: usize) -> Vec<u8> {
    let state_heads = (STATE_GLOBAL_HEADS / width) as i64;
    let state_bytes = state_heads * STATE_HEAD_BYTES as i64;
    let lcm = BLOCK_BYTES as i64;
    let num_lcm = (device_lcm_pages - 1) as i64;
    let payload = serde_json::json!({
        "plan": {
            "prefix_granularity": PAGE,
            "lcm_block_bytes": lcm,
            "num_lcm_blocks": num_lcm,
            "groups": [
                {
                    "group_id": "full",
                    "cache_blocks_per_lcm_block": 1,
                    "page_count": device_lcm_pages as i64
                },
                {
                    "group_id": "state",
                    "cache_blocks_per_lcm_block": 1,
                    "page_count": device_lcm_pages as i64
                }
            ],
            "planes": [
                { "plane_id": "kv", "bytes_per_lcm_block": lcm, "arena_offset_bytes": 0 },
                { "plane_id": "st", "bytes_per_lcm_block": state_bytes, "arena_offset_bytes": num_lcm * lcm }
            ],
            "fields": [
                {
                    "group_id": "full",
                    "field_id": "layer.0.kv",
                    "plane_id": "kv",
                    "shape": [PAGE as i64, (BLOCK_BYTES / PAGE as usize) as i64],
                    "dtype": "uint8",
                    "field_offset_bytes": 0,
                    "page_stride_bytes": BLOCK_BYTES as i64
                },
                {
                    "group_id": "state",
                    "field_id": "layer.0.st",
                    "plane_id": "st",
                    "shape": [state_heads, STATE_HEAD_BYTES as i64],
                    "dtype": "uint8",
                    "field_offset_bytes": 0,
                    "page_stride_bytes": state_bytes
                }
            ]
        },
        "group_specs": [
            {
                "group_id": "full",
                "attn_kind": "full_history",
                "rows_per_page": PAGE as i64,
                "entry_stride_tokens": 1,
                "replayable": false,
                "transfer_policy": "full_suffix",
                "shard_count": width as i64
            },
            {
                "group_id": "state",
                "attn_kind": "state",
                "replayable": false,
                "transfer_policy": "latest_snapshot",
                "shard_count": 1
            }
        ],
        "transfer_schema": {
            "fields": [
                {
                    "field_id": "layer.0.st",
                    "partition": {
                        "axis": 0,
                        "global_extent": STATE_GLOBAL_HEADS as i64,
                        "global_parts": []
                    }
                }
            ]
        }
    });
    serde_json::to_vec(&payload).unwrap()
}

fn make_scheduler(
    name: &str,
    incarnation: u64,
    width: usize,
    device_lcm_pages: usize,
    with_disk: bool,
    versioned_shards: bool,
) -> Scheduler {
    interpreter();
    let (snap_pages, max_retract) = if width == 8 {
        (1, 0)
    } else {
        (device_lcm_pages, 8)
    };
    let tiers = if with_disk {
        format!(
            "cfg.num_device_pages = {device_lcm_pages}\n\
             cfg.num_snapshot_pages = {snap_pages}\n\
             cfg.max_retracted_requests = {max_retract}\n\
             cfg.disable_l2_cache = False\n\
             cfg.num_host_pages = {host_pages}\n\
             cfg.host_group_packing = [{width}, 1]\n\
             cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack\n\
             cfg.num_disk_pages = {disk_pages}\n\
             cfg.disk_config = ts.DiskConfig(max_operations=4, max_blocks_per_operation=64,\n\
                                             max_host_blocks={max_host_blocks}, max_write_host_blocks={max_write_blocks},\n\
                                             max_queued_cleanings=64, min_prefetch_tokens=1,\n\
                                             wait_timeout_us=10_000_000, ready_timeout_us=10_000_000,\n\
                                             min_clean_tokens=0, require_reuse_for_cleaning=False)",
            host_pages = HOST_LCM_BLOCKS + 1,
            disk_pages = DISK_LCM_BLOCKS + 1,
            max_host_blocks = 4 * width,
            max_write_blocks = 2 * width,
        )
    } else {
        format!(
            "cfg.num_device_pages = {device_lcm_pages}\n\
             cfg.num_snapshot_pages = 1\n\
             cfg.disable_l2_cache = True"
        )
    };
    let vs = if versioned_shards { "True" } else { "False" };
    let role = if width == 8 { "P" } else { "Fused" };
    Python::attach(|py| {
        let code = format!(
            r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.{role}
cfg.prefix_granularity = {PAGE}
cfg.max_scheduled_tokens = 256
cfg.max_batch_size = 8
cfg.max_resident_requests = 8
{tiers}
cfg.cache_groups = [
    ts.CacheGroupConfig(
        group_id="full",
        block_granularity={PAGE},
        total_pages={device_lcm_pages},
        attn_kind=ts.AttnKind.Full,
        cache_blocks_per_lcm_block={width},
        transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        shard_count={width},
    ),
    ts.CacheGroupConfig(
        group_id="state",
        block_granularity=0,
        total_pages={device_lcm_pages},
        attn_kind=ts.AttnKind.State,
        cache_blocks_per_lcm_block=1,
        transfer_policy=ts.CacheTransferPolicy.LatestSnapshot,
        shard_count=1,
    ),
]
cfg.peer_config = ts.PeerConfig(
    name="{name}",
    incarnation={incarnation},
    max_leases=16,
    soft_pin_us=2_000_000,
    hard_pin_us=10_000_000,
    lookup_timeout_us=2_000_000,
    min_fetch_tokens={PAGE},
    landing_tier=ts.PeerLandingTier.Device,
    max_host_landing_blocks=0,
    host_ready_timeout_us=0,
    versioned_shards={vs},
)
result = ts.Scheduler(cfg)
"#
        );
        let globals = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    })
}

fn host_layout(num_lcm_blocks: usize, width: usize, rank: usize) -> HostLayout {
    HostLayout {
        groups: vec![
            GroupLayout {
                packing: 1,
                block_bytes: BLOCK_BYTES,
                shard_count: width,
            },
            GroupLayout {
                packing: 1,
                block_bytes: BLOCK_BYTES,
                shard_count: 1,
            },
        ],
        lcm_block_bytes: BLOCK_BYTES,
        num_lcm_blocks,
        owner_rank: rank,
        tp_rank: rank,
    }
}

struct ScratchFile(PathBuf);
impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Fault-injectable writer copying out of a holder's staging ring into the shared test arena.
struct FaultableArenaWriter {
    ring: Arc<Memory>,
    arena: Arena,
    fail_writes: Arc<AtomicBool>,
}

impl PageWriter for FaultableArenaWriter {
    fn write(&self, _session: &str, spans: &Spans) -> Result<(), String> {
        if self.fail_writes.load(Ordering::Acquire) {
            return Err("injected RDMA write fault".into());
        }
        for ((&source, &destination), &length) in spans
            .sources
            .iter()
            .zip(&spans.destinations)
            .zip(&spans.lengths)
        {
            let mut bytes = vec![0u8; length as usize];
            self.ring
                .read_at(source as usize - self.ring.address(), &mut bytes);
            self.arena.write(destination, &bytes);
        }
        Ok(())
    }
}

fn peer_settings(name: &str, incarnation: u64, push_us: i64, quarantine_us: i64) -> PeerSettings {
    PeerSettings {
        name: name.to_owned(),
        incarnation,
        deployment_id: "deployment".into(),
        weight_version: "v1".into(),
        timeouts: Timeouts {
            answer_us: 2_000_000,
            quarantine_us,
            push_us,
        },
        record_lifetime: Duration::from_secs(10),
    }
}

struct RankState {
    rank: usize,
    kv_dev_base: u64,
    st_dev_base: u64,
    scheduler: Arc<Mutex<Scheduler>>,
    peer: Peer,
    disk: Option<(DiskTier, Memory, ScratchFile, HostLayout)>,
    acks: Vec<CacheEvent>,
    in_flight: HashMap<u64, DiskOpView>,
}

struct MultiRankReplica {
    name: String,
    width: usize,
    ranks: Vec<RankState>,
    /// Expected bytes written to Disk per `(rank, group, disk_block)`.
    on_disk: HashMap<(usize, u32, i32), Vec<u8>>,
    /// Fault flag shared across all holder ranks' PageWriters.
    fail_writes: Arc<AtomicBool>,
    next_seed: u8,
    /// Last LookupAnswered offered pages observed on rank 0.
    last_offered_pages: Vec<PeerPageView>,
    /// Whether rank 0 observed `FetchFrozen`.
    saw_frozen: bool,
}

fn req_spec(request_id: &str, tokens: Vec<i32>, peer_holder: &str) -> RequestSpec {
    RequestSpec {
        admission_id: 0,
        request_id: request_id.to_owned(),
        session_id: String::new(),
        cache_group_id: String::new(),
        shared_prefix_tokens: Vec::new(),
        peer_holder: peer_holder.to_owned(),
        tokens,
        max_new_tokens: 1,
        max_cached_prefix_tokens: i32::MAX,
    }
}

fn fwd_event(kind: ForwardEventKind, request_id: &str, tokens: Vec<i32>) -> ForwardEvent {
    ForwardEvent {
        kind,
        request_id: request_id.to_owned(),
        tokens,
        spec_candidate_ids: Vec::new(),
        reserve_num_tokens: 0,
        bootstrap_token: -1,
    }
}

impl MultiRankReplica {
    #[allow(clippy::too_many_arguments)]
    fn start(
        width: usize,
        incarnation: u64,
        device_lcm_pages: usize,
        with_disk: bool,
        versioned_shards: bool,
        arena_base: u64,
        arena: &Arena,
        semantic: [u8; 32],
    ) -> Self {
        Self::start_with_options(
            width,
            incarnation,
            device_lcm_pages,
            with_disk,
            with_disk,
            versioned_shards,
            arena_base,
            arena,
            semantic,
            5_000_000,
            10_000,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn start_with_push_us(
        width: usize,
        incarnation: u64,
        device_lcm_pages: usize,
        with_disk: bool,
        versioned_shards: bool,
        arena_base: u64,
        arena: &Arena,
        semantic: [u8; 32],
        push_us: i64,
    ) -> Self {
        Self::start_with_options(
            width,
            incarnation,
            device_lcm_pages,
            with_disk,
            with_disk,
            versioned_shards,
            arena_base,
            arena,
            semantic,
            push_us,
            10_000,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn start_with_options(
        width: usize,
        incarnation: u64,
        device_lcm_pages: usize,
        with_disk: bool,
        advertise_disk_push: bool,
        versioned_shards: bool,
        arena_base: u64,
        arena: &Arena,
        semantic: [u8; 32],
        push_us: i64,
        quarantine_us: i64,
        custom_pusher: Option<Arc<dyn PagePusher>>,
    ) -> Self {
        let mut listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let name = format!("127.0.0.1:{}", listener.port());
        let fail_writes = Arc::new(AtomicBool::new(false));
        let state_bytes = (STATE_GLOBAL_HEADS / width) * STATE_HEAD_BYTES;
        let stride_per_rank = 0x80_000u64; // 512 KiB per rank inside the Arena

        let mut push_eps = Vec::with_capacity(width);
        let mut disks = Vec::with_capacity(width);
        let mut geometries = Vec::with_capacity(width);
        let mut member_blobs = Vec::with_capacity(width);

        for rank in 0..width {
            let push = PushEndpoint::bind(
                "127.0.0.1",
                0,
                "127.0.0.1",
                incarnation * 16 + rank as u64 + 1,
            )
            .unwrap();
            let kv_dev_base = arena_base + (rank as u64) * stride_per_rank;
            let st_dev_base = kv_dev_base + 0x40_000;
            let region_len = stride_per_rank;

            let (disk_entry, disk_bounds) = if with_disk {
                let dir = std::env::var_os("TS_DISK_TEST_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir);
                let fid = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
                let file = ScratchFile(dir.join(format!(
                    "ts-peer-v1-disk-{}-{}-r{}.bin",
                    std::process::id(),
                    fid,
                    rank
                )));
                let host = Memory::map(HOST_LCM_BLOCKS * BLOCK_BYTES).unwrap();
                let h_layout = host_layout(HOST_LCM_BLOCKS, width, rank);
                let d_layout = host_layout(DISK_LCM_BLOCKS, width, rank);
                let tier = DiskTier::start(
                    &DiskSettings {
                        file: file.0.clone(),
                        file_bytes: (DISK_LCM_BLOCKS * BLOCK_BYTES) as u64,
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
                        host_layout: h_layout.clone(),
                        disk_layout: d_layout.clone(),
                        max_operations: 4,
                        push_ring_bytes: 4 * BLOCK_BYTES,
                    },
                    None,
                )
                .unwrap();
                let (executor, disk_layout, ring, slot_bytes) = tier.push_source().unwrap();
                let bounds: Vec<i64> = if advertise_disk_push {
                    disk_layout
                        .groups
                        .iter()
                        .map(|g| {
                            (1 + disk_layout.num_lcm_blocks * g.packing * g.shard_count) as i64
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                if advertise_disk_push {
                    push.serve(PushSource {
                        executor,
                        disk_layout,
                        writer: Arc::new(FaultableArenaWriter {
                            ring: ring.clone(),
                            arena: arena.clone(),
                            fail_writes: fail_writes.clone(),
                        }),
                        ring,
                        slot_bytes,
                    })
                    .unwrap();
                }
                (Some((tier, host, file, h_layout)), bounds)
            } else {
                (None, Vec::new())
            };

            let geom = RankGeometry {
                session: format!("sess-{name}-r{rank}"),
                device: vec![
                    DeviceField {
                        group_id: 0,
                        field_id: "layer.0.kv".into(),
                        address: kv_dev_base,
                        page_stride: BLOCK_BYTES as u64,
                        bytes: BLOCK_BYTES as u64,
                    },
                    DeviceField {
                        group_id: 1,
                        field_id: "layer.0.st".into(),
                        address: st_dev_base,
                        page_stride: state_bytes as u64,
                        bytes: state_bytes as u64,
                    },
                ],
                host: Some(HostPool {
                    address: kv_dev_base + 0x60_000,
                    groups: vec![
                        HostGroup {
                            parent_bytes: BLOCK_BYTES as u64,
                            packing: 1,
                            block_bytes: BLOCK_BYTES as u64,
                            field_offsets: vec![("layer.0.kv".into(), 0)],
                        },
                        HostGroup {
                            parent_bytes: BLOCK_BYTES as u64,
                            packing: 1,
                            block_bytes: BLOCK_BYTES as u64,
                            field_offsets: vec![("layer.0.st".into(), 0)],
                        },
                    ],
                }),
                push: Some(push.name().to_owned()),
            };
            let host_bounds = vec![
                (1 + HOST_LCM_BLOCKS * width) as i64,
                (1 + HOST_LCM_BLOCKS) as i64,
            ];
            let member = Member {
                geometry: geom.clone(),
                bounds: host_bounds,
                disk_bounds,
                regions: vec![(kv_dev_base, region_len)],
            };
            member_blobs.push(member.encode(rank));
            geometries.push(geom);
            disks.push(disk_entry);
            push_eps.push(Arc::new(push));
        }

        let contract_wire = hybrid_contract_wire(width, device_lcm_pages);
        let registered = Arc::new(
            RegisteredReplica::register(semantic, &contract_wire, &member_blobs)
                .expect("registered replica"),
        );
        listener.set_registered_layout(registered.clone());
        let v1_channel = listener.versioned_channel();

        let mut ranks = Vec::with_capacity(width);
        let mut listener_opt = Some(listener);
        let mut channel_opt = Some(v1_channel);
        for (rank, (disk, push)) in disks.into_iter().zip(push_eps).enumerate() {
            let sched = Arc::new(Mutex::new(make_scheduler(
                &name,
                incarnation,
                width,
                device_lcm_pages,
                with_disk,
                versioned_shards,
            )));
            let pusher: Arc<dyn PagePusher> = custom_pusher.clone().unwrap_or(push);
            let mut peer = Peer::start(
                peer_settings(&name, incarnation, push_us, quarantine_us),
                PeerProgress::Coordinated,
                rank,
                geometries.clone(),
                listener_opt.take(),
                sched.clone(),
                PeerTransfers {
                    reader: Arc::new(arena.clone()),
                    pusher,
                },
            )
            .unwrap();
            peer.enable_versioned(
                registered.clone(),
                name.clone(),
                incarnation,
                channel_opt.take(),
                0,
            );
            let kv_dev_base = arena_base + (rank as u64) * stride_per_rank;
            let st_dev_base = kv_dev_base + 0x40_000;
            ranks.push(RankState {
                rank,
                kv_dev_base,
                st_dev_base,
                scheduler: sched,
                peer,
                disk,
                acks: Vec::new(),
                in_flight: HashMap::new(),
            });
        }

        Self {
            name,
            width,
            ranks,
            on_disk: HashMap::new(),
            fail_writes,
            next_seed: 1,
            last_offered_pages: Vec::new(),
            saw_frozen: false,
        }
    }

    fn submit_requests(&mut self, specs: &[RequestSpec]) {
        for r in &mut self.ranks {
            r.scheduler.lock().submit_requests(specs).unwrap();
        }
        if self.width == 8 {
            let bootstrapped: Vec<ForwardEvent> = specs
                .iter()
                .map(|s| fwd_event(ForwardEventKind::PdBootstrapped, &s.request_id, Vec::new()))
                .collect();
            self.advance_forward(&bootstrapped);
        }
    }

    fn finish_request(&mut self, request_id: &str) {
        if self.width == 8 {
            self.advance_forward(&[fwd_event(
                ForwardEventKind::ExtendResult,
                request_id,
                vec![999],
            )]);
            self.round();
            self.advance_forward(&[fwd_event(
                ForwardEventKind::PdSucceeded,
                request_id,
                Vec::new(),
            )]);
        } else {
            self.advance_forward(&[
                fwd_event(ForwardEventKind::ExtendResult, request_id, vec![999]),
                fwd_event(ForwardEventKind::Finish, request_id, Vec::new()),
            ]);
        }
        self.round();
    }

    fn advance_forward(&mut self, events: &[ForwardEvent]) {
        for r in &mut self.ranks {
            r.scheduler.lock().advance(events).unwrap();
        }
    }

    /// Run one coordinated control round across all ranks of this replica.
    fn round(&mut self) -> (Vec<(String, i32)>, Vec<PeerFetchView>) {
        // 1. Disk tier collective agreement + advance_head across all ranks.
        if self.ranks[0].disk.is_some() {
            let reports: Vec<Vec<u8>> = self
                .ranks
                .iter_mut()
                .map(|r| r.disk.as_mut().unwrap().0.report())
                .collect();
            for r in &mut self.ranks {
                let (disk, host, _, h_layout) = r.disk.as_mut().unwrap();
                let storage = disk.agree(&reports, true, true).unwrap();
                for done in &storage.disk_done {
                    assert!(done.success, "Disk op {} failed", done.op_id);
                    let op = r.in_flight.remove(&done.op_id).expect("in-flight disk op");
                    if op.kind == DiskOpKind::Write {
                        // Wipe Host copy once persisted on Disk to prove bytes come from Disk.
                        for row in 0..op.group_ids.len() {
                            if let Some(range) = h_layout
                                .owned_range(op.group_ids[row], op.host_blocks[row])
                                .unwrap()
                            {
                                host.write_at(range.offset, &vec![0u8; range.size]);
                            }
                        }
                    }
                }
                let acks = std::mem::take(&mut r.acks);
                r.scheduler.lock().advance_head(&storage, &acks).unwrap();
            }
        }

        // 2. Peer collective agreement (`agree_with_scheduler`) across all ranks.
        let peer_reports: Vec<Vec<u8>> = self.ranks.iter_mut().map(|r| r.peer.report()).collect();
        let mut r0_rows = Vec::new();
        let mut r0_fetches = Vec::new();
        let seed_base = self.next_seed;
        let mut wrote_any_disk = false;

        for r in &mut self.ranks {
            let mut sched = r.scheduler.lock();
            let round = r
                .peer
                .agree_with_scheduler(&peer_reports, &mut sched)
                .unwrap();
            if r.rank == 0 {
                for ev in &round.events {
                    if ev.kind == PeerEventKind::LookupAnswered && ev.num_tokens > 0 {
                        self.last_offered_pages = ev.pages.clone();
                    }
                    if ev.kind == PeerEventKind::FetchFrozen {
                        self.saw_frozen = true;
                    }
                }
            }
            sched.advance_peer(round.now_us, &round.events).unwrap();
            let plan = sched.next_execution_plan(true, false);
            let fetches = plan.peer_fetches();
            let lookups = sched.take_peer_lookups();
            drop(sched);

            if let Some((disk, host, _, h_layout)) = &mut r.disk {
                for op in plan.disk_ops() {
                    if op.kind == DiskOpKind::Write {
                        wrote_any_disk = true;
                        for row in 0..op.group_ids.len() {
                            let group = op.group_ids[row];
                            let hblk = op.host_blocks[row];
                            let dblk = op.disk_blocks[row];
                            if let Some(range) = h_layout.owned_range(group, hblk).unwrap() {
                                let mut page = vec![0u8; range.size];
                                if group == 1 && self.width == 1 {
                                    // TP1 State block: 8 distinct 512-byte head slices so
                                    // TP8 reader slicing is verified byte-for-byte per head!
                                    for head in 0..STATE_GLOBAL_HEADS {
                                        let fill = seed_base
                                            .wrapping_add(dblk as u8)
                                            .wrapping_add((head as u8 + 1) * 19)
                                            | 1;
                                        let lo = head * STATE_HEAD_BYTES;
                                        for (k, b) in
                                            page[lo..lo + STATE_HEAD_BYTES].iter_mut().enumerate()
                                        {
                                            *b = fill.wrapping_add((k & 0x3f) as u8);
                                        }
                                    }
                                } else {
                                    // Distinct per (rank, group, disk_block, byte_idx).
                                    let tag = seed_base
                                        .wrapping_add((r.rank as u8 + 1) * 29)
                                        .wrapping_add((group as u8 + 1) * 7)
                                        .wrapping_add(dblk as u8)
                                        | 1;
                                    for (k, b) in page.iter_mut().enumerate() {
                                        *b = tag.wrapping_add(((k * 3) & 0x7f) as u8);
                                    }
                                }
                                host.write_at(range.offset, &page);
                                self.on_disk.insert((r.rank, group, dblk), page);
                            }
                        }
                    }
                    r.in_flight.insert(op.op_id, op);
                }
                disk.submit(&plan).unwrap();
                r.acks = plan
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

            r.peer.dispatch(fetches.clone(), lookups, round.now_us);
            if plan.needs_device() && !fetches.is_empty() {
                r.peer.fence().ready();
            }

            if r.rank == 0 {
                r0_fetches = fetches;
                r0_rows = plan
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
            }
        }
        if wrote_any_disk {
            self.next_seed = self.next_seed.wrapping_add(13).max(1);
        }
        (r0_rows, r0_fetches)
    }

    fn run_to_finish(&mut self, request_id: &str, tokens: Vec<i32>) {
        self.submit_requests(&[req_spec(request_id, tokens, "")]);
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            assert!(
                Instant::now() < deadline,
                "{request_id} did not prefill on holder"
            );
            let (rows, _) = self.round();
            if rows.iter().any(|(r, _)| r == request_id) {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        self.finish_request(request_id);
    }

    fn drain_disk_cleanings(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            self.round();
            let stats = self.ranks[0].scheduler.lock().disk_stats();
            let busy = self
                .ranks
                .iter()
                .any(|r| !r.in_flight.is_empty() || !r.acks.is_empty())
                || stats.queued_cleanings > 0;
            if !busy {
                return;
            }
            assert!(Instant::now() < deadline, "Disk cleaning did not complete");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    fn evict_all_device_and_host(&mut self, seed_base: i32, pages_per_req: i32, count: i32) {
        for i in 0..count {
            let start = seed_base + i * 1000;
            let tokens: Vec<i32> = (start..start + pages_per_req * PAGE).collect();
            self.run_to_finish(&format!("evict-{seed_base}-{i}"), tokens);
            self.drain_disk_cleanings();
        }
        assert!(
            self.ranks[0].scheduler.lock().disk_stats().disk_blocks > 0 && !self.on_disk.is_empty(),
            "expected cleaned Disk blocks on holder"
        );
    }

    fn assert_no_leaked_pins(&self) {
        for r in &self.ranks {
            let stats = r.scheduler.lock().peer_stats();
            assert_eq!(
                (stats.leases, stats.pinned_blocks, stats.landing_requests),
                (0, 0, 0),
                "leaked peer state on {} rank {}: {stats:?}",
                self.name,
                r.rank
            );
        }
    }
}

fn drive_peer_landing(
    holder: &mut MultiRankReplica,
    reader: &mut MultiRankReplica,
    request_id: &str,
    prompt: Vec<i32>,
) -> (i32, PeerFetchView) {
    reader.submit_requests(&[req_spec(request_id, prompt, &holder.name)]);
    let mut observed_fetch = None;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        assert!(
            Instant::now() < deadline,
            "{request_id} did not complete peer landing"
        );
        holder.round();
        let (rows, fetches) = reader.round();
        if let Some(f) = fetches.into_iter().find(|f| f.request_id == request_id) {
            observed_fetch = Some(f);
        }
        if let Some((_, prefix)) = rows.into_iter().find(|(r, _)| r == request_id) {
            // Pump a couple more rounds so holder finishes Release ACK & cleanup.
            for _ in 0..5 {
                holder.round();
                reader.round();
                std::thread::sleep(Duration::from_millis(2));
            }
            return (prefix, observed_fetch.expect("expected peer fetch"));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn verify_landed_bytes(
    holder: &MultiRankReplica,
    reader: &MultiRankReplica,
    arena: &Arena,
    offered: &[PeerPageView],
    fetch: &PeerFetchView,
) {
    let by_slot: HashMap<(u32, i32), PeerPageView> =
        offered.iter().map(|p| ((p.group_id, p.slot), *p)).collect();
    let mut disk_pages_checked = 0usize;
    for dst in &fetch.destinations {
        let src = by_slot[&(dst.group_id, dst.slot)];
        if src.tier != PeerTier::Disk {
            continue;
        }
        disk_pages_checked += 1;
        if dst.group_id == 0 {
            // KV full_history: reader rank = (dst.block_id - 1) % 8,
            // holder rank = (src.block_id - 1) % holder.width.
            let reader_rank = ((dst.block_id - 1) as usize) % 8;
            let local_dst = 1 + ((dst.block_id - 1) as u64) / 8;
            let holder_rank = ((src.block_id - 1) as usize) % holder.width;
            let expected = &holder.on_disk[&(holder_rank, 0, src.block_id)];
            let dst_addr = reader.ranks[reader_rank].kv_dev_base + local_dst * BLOCK_BYTES as u64;
            let actual = arena.read_bytes(dst_addr, BLOCK_BYTES);
            assert_eq!(
                actual, *expected,
                "KV Disk byte mismatch slot={} src_blk={} (h_rank={}) -> dst_blk={} (r_rank={})",
                dst.slot, src.block_id, holder_rank, dst.block_id, reader_rank
            );
        } else if dst.group_id == 1 {
            // State checkpoint: replicated across all 8 reader ranks with TP8 head slicing.
            for reader_rank in 0..8 {
                let dst_addr = reader.ranks[reader_rank].st_dev_base
                    + (dst.block_id as u64) * STATE_HEAD_BYTES as u64;
                let actual = arena.read_bytes(dst_addr, STATE_HEAD_BYTES);
                let expected = if holder.width == 1 {
                    let full_state = &holder.on_disk[&(0, 1, src.block_id)];
                    let lo = reader_rank * STATE_HEAD_BYTES;
                    full_state[lo..lo + STATE_HEAD_BYTES].to_vec()
                } else {
                    holder.on_disk[&(reader_rank, 1, src.block_id)][..STATE_HEAD_BYTES].to_vec()
                };
                assert_eq!(
                    actual, expected,
                    "State Disk byte mismatch src_blk={} dst_blk={} reader_rank={}",
                    src.block_id, dst.block_id, reader_rank
                );
            }
        }
    }
    assert!(
        disk_pages_checked > 0,
        "expected at least one Disk page in offered pages: {offered:?}"
    );
}

#[test]
fn tp8_kvp8_holder_disk_to_tp8_kvp8_reader_hybrid() {
    let semantic = [0xA5; 32];
    let arena = Arena::new(ARENA_BASE_OFFSET as usize + 16 * 0x80_000);
    // TP8 holder: 5 LCM device pages (4 usable -> 32 KV blocks, 4 State blocks),
    // 8 Host LCM blocks, 16 Disk LCM blocks.
    let mut holder = MultiRankReplica::start(
        8,
        1,
        HOLDER_DEVICE_LCM,
        true,
        true,
        ARENA_BASE_OFFSET,
        &arena,
        semantic,
    );
    let mut reader = MultiRankReplica::start(
        8,
        2,
        READER_DEVICE_LCM,
        false,
        true,
        ARENA_BASE_OFFSET + 8 * 0x80_000,
        &arena,
        semantic,
    );

    // Warm an 8-page (32-token) prompt so ALL 8 KVP8 ranks own Disk KV blocks + State checkpoint.
    let prefix: Vec<i32> = (1..=8 * PAGE).collect();
    holder.run_to_finish("seed", prefix.clone());
    // Push seed out of Device (4 state blocks) and Host (8 state blocks) onto Disk!
    holder.evict_all_device_and_host(10_000, 8, 12);

    // Reader requests `prefix` + 1 new page from TP8 holder.
    let mut prompt = prefix.clone();
    prompt.extend(90_000..90_000 + PAGE);
    let (landed_prefix, fetch) =
        drive_peer_landing(&mut holder, &mut reader, "tp8-disk-hit", prompt);

    assert_eq!(landed_prefix, prefix.len() as i32);
    let offered = reader.last_offered_pages.clone();
    assert!(
        offered
            .iter()
            .any(|p| p.group_id == 0 && p.tier == PeerTier::Disk),
        "expected Disk KV pages in offer: {offered:?}"
    );
    assert!(
        offered
            .iter()
            .any(|p| p.group_id == 1 && p.tier == PeerTier::Disk),
        "expected Disk State page in offer: {offered:?}"
    );
    verify_landed_bytes(&holder, &reader, &arena, &offered, &fetch);

    // Finish reader request and verify zero leaked pins or leases on both sides.
    reader.finish_request("tp8-disk-hit");
    holder.assert_no_leaked_pins();
    reader.assert_no_leaked_pins();
}

#[test]
fn tp1_kvp1_holder_disk_to_tp8_kvp8_reader_hybrid_cross_layout() {
    let semantic = [0x5A; 32];
    let arena = Arena::new(ARENA_BASE_OFFSET as usize + 10 * 0x80_000);
    // TP1/KVP1 Decode holder: 11 LCM device pages (10 usable KV/State blocks),
    // plus 8 Host LCM blocks and 16 Disk LCM blocks.
    let mut holder =
        MultiRankReplica::start(1, 10, 11, true, false, ARENA_BASE_OFFSET, &arena, semantic);
    let mut reader = MultiRankReplica::start(
        8,
        11,
        READER_DEVICE_LCM,
        false,
        true,
        ARENA_BASE_OFFSET + 0x80_000,
        &arena,
        semantic,
    );

    // Warm a 4-page (16-token) prompt on TP1 holder, then evict out of Device & Host onto Disk.
    let prefix: Vec<i32> = (1..=4 * PAGE).collect();
    holder.run_to_finish("d-seed", prefix.clone());
    holder.evict_all_device_and_host(20_000, 4, 6);

    let mut prompt = prefix.clone();
    prompt.extend(80_000..80_000 + PAGE);
    let (landed_prefix, fetch) =
        drive_peer_landing(&mut holder, &mut reader, "tp1-to-tp8-disk", prompt);

    assert_eq!(landed_prefix, prefix.len() as i32);
    let offered = reader.last_offered_pages.clone();
    assert!(
        offered
            .iter()
            .any(|p| p.group_id == 0 && p.tier == PeerTier::Disk),
        "expected Disk KV pages in offer: {offered:?}"
    );
    assert!(
        offered
            .iter()
            .any(|p| p.group_id == 1 && p.tier == PeerTier::Disk),
        "expected Disk State page in offer: {offered:?}"
    );
    // Verifies BOTH KVP8 KV routing AND per-head TP8 slicing of TP1's State Disk block!
    verify_landed_bytes(&holder, &reader, &arena, &offered, &fetch);

    reader.finish_request("tp1-to-tp8-disk");
    holder.assert_no_leaked_pins();
    reader.assert_no_leaked_pins();
}

#[test]
fn v1_disk_push_write_fault_quarantines_then_unfreezes_and_releases_holder() {
    let semantic = [0x3C; 32];
    let arena = Arena::new(ARENA_BASE_OFFSET as usize + 10 * 0x80_000);
    let mut holder =
        MultiRankReplica::start(1, 20, 11, true, false, ARENA_BASE_OFFSET, &arena, semantic);
    let quarantine_us = 250_000;
    let mut reader = MultiRankReplica::start_with_options(
        8,
        21,
        READER_DEVICE_LCM,
        false,
        false,
        true,
        ARENA_BASE_OFFSET + 0x80_000,
        &arena,
        semantic,
        10_000_000,
        quarantine_us,
        None,
    );

    let prefix: Vec<i32> = (1..=4 * PAGE).collect();
    holder.run_to_finish("fault-seed", prefix.clone());
    holder.evict_all_device_and_host(30_000, 4, 6);

    // Inject RDMA write fault on the holder's push writer -> Message::Unknown ->
    // ReadRetirement::Unknown -> reader emits FetchFrozen and holds both reader
    // targets and holder pins through `quarantine_us`, then emits TargetRetired
    // + Release(Abort) so neither side leaks KV blocks.
    holder.fail_writes.store(true, Ordering::Release);
    let mut prompt = prefix.clone();
    prompt.extend(70_000..70_000 + PAGE);
    reader.submit_requests(&[req_spec("fault-req", prompt, &holder.name)]);

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        holder.round();
        let (rows, _) = reader.round();
        if reader.saw_frozen {
            if let Some((_, landed)) = rows.into_iter().find(|(r, _)| r == "fault-req") {
                assert_eq!(
                    landed, 0,
                    "frozen request must fall back with 0 reused tokens"
                );
                break;
            }
        }
        assert!(
            Instant::now() < deadline,
            "expected FetchFrozen and local fallback after write fault"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    // Immediately after fallback (still inside the 250ms quarantine window),
    // reader targets stay frozen and holder Disk blocks remain pinned.
    assert!(
        holder.ranks[0].scheduler.lock().peer_stats().pinned_blocks > 0,
        "unknown write retirement must keep holder Disk blocks pinned during quarantine"
    );
    for r in &reader.ranks {
        let stats = r.scheduler.lock().peer_stats();
        assert_eq!(stats.frozen_targets, 1, "rank {} frozen_targets", r.rank);
        assert!(stats.frozen_blocks > 0, "rank {} frozen_blocks", r.rank);
        assert_eq!(stats.frozen_landings, 1, "rank {} frozen_landings", r.rank);
    }

    // Pump rounds past `quarantine_us`: reader must emit TargetRetired and
    // Release(Abort), unfreezing all 8 reader ranks and releasing holder pins.
    let unfreeze_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        holder.round();
        reader.round();
        let all_unfrozen = reader.ranks.iter().all(|r| {
            let st = r.scheduler.lock().peer_stats();
            st.frozen_targets == 0 && st.frozen_blocks == 0
        });
        let holder_clear = holder.ranks[0].scheduler.lock().peer_stats().pinned_blocks == 0;
        if all_unfrozen && holder_clear {
            break;
        }
        assert!(
            Instant::now() < unfreeze_deadline,
            "expected quarantine expiry to unfreeze reader targets and release holder pins"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    reader.finish_request("fault-req");
    holder.assert_no_leaked_pins();
    reader.assert_no_leaked_pins();
}

#[test]
fn dead_holder_unanswered_cancel_unfreezes_reader_after_budget_and_quarantine() {
    /// Emulates a holder whose push endpoint goes completely silent (crashed /
    /// partitioned / restarted on a new port): never answers Push or Cancel,
    /// triggers early `cancelling` freeze after `freeze_wait`, then expires the
    /// retry budget with `PushOutcome::Unknown`.
    struct SilentHolderPusher {
        freeze_wait: Duration,
        retry_budget: Duration,
    }
    impl PagePusher for SilentHolderPusher {
        fn push(&self, _: &str, _: &str, _: Duration, _: Vec<PushItem>) -> Result<(), String> {
            Err("unused in v1".into())
        }
        fn start_push_many(
            &self,
            _targets: Vec<PushTarget>,
            budget: Duration,
            _submit_until: Instant,
            sealed: Arc<AtomicBool>,
            cancelling: Arc<AtomicBool>,
            finish: Box<dyn FnOnce(PushOutcome) + Send>,
        ) {
            let freeze_wait = self.freeze_wait;
            let retry_budget = self.retry_budget;
            std::thread::spawn(move || {
                let wait_until = Instant::now() + budget;
                while !sealed.load(Ordering::Acquire) && Instant::now() < wait_until {
                    std::thread::sleep(Duration::from_millis(5));
                }
                std::thread::sleep(freeze_wait);
                cancelling.store(true, Ordering::Release);
                std::thread::sleep(retry_budget);
                finish(PushOutcome::Unknown(
                    "holder cancel retries exhausted".into(),
                ));
            });
        }
    }

    let semantic = [0x4D; 32];
    let arena = Arena::new(ARENA_BASE_OFFSET as usize + 10 * 0x80_000);
    let mut holder =
        MultiRankReplica::start(1, 50, 11, true, false, ARENA_BASE_OFFSET, &arena, semantic);
    let silent_pusher = Arc::new(SilentHolderPusher {
        freeze_wait: Duration::from_millis(30),
        retry_budget: Duration::from_millis(120),
    });
    let mut reader = MultiRankReplica::start_with_options(
        8,
        51,
        READER_DEVICE_LCM,
        false,
        false,
        true,
        ARENA_BASE_OFFSET + 0x80_000,
        &arena,
        semantic,
        40_000,  // 40ms push timeout before entering cancel phase
        150_000, // 150ms quarantine after Unknown retirement
        Some(silent_pusher),
    );

    let prefix: Vec<i32> = (1..=4 * PAGE).collect();
    holder.run_to_finish("silent-seed", prefix.clone());
    holder.evict_all_device_and_host(55_000, 4, 6);

    let mut prompt = prefix.clone();
    prompt.extend(75_000..75_000 + PAGE);
    reader.submit_requests(&[req_spec("silent-req", prompt, &holder.name)]);

    // Drive until early FetchFrozen fallback completes locally.
    let freeze_deadline = Instant::now() + Duration::from_secs(10);
    loop {
        holder.round();
        let (rows, _) = reader.round();
        if reader.saw_frozen {
            if let Some((_, landed)) = rows.into_iter().find(|(r, _)| r == "silent-req") {
                assert_eq!(landed, 0, "early-frozen fallback must reuse 0 tokens");
                break;
            }
        }
        assert!(
            Instant::now() < freeze_deadline,
            "expected early FetchFrozen fallback while holder is silent"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    for r in &reader.ranks {
        let stats = r.scheduler.lock().peer_stats();
        assert_eq!(
            stats.frozen_targets, 1,
            "rank {} frozen during retry",
            r.rank
        );
        assert!(stats.frozen_blocks > 0, "rank {} frozen_blocks", r.rank);
    }

    // Stop pumping `holder` completely to simulate a dead/unreachable holder
    // node: even with ZERO responses to Cancel or Release, the reader must
    // unfreeze all 8 ranks once retry_budget + quarantine_us elapses (in
    // production: 60s cancel retry + 5s quarantine ~= 65s < 70s).
    reader.finish_request("silent-req");
    let unfreeze_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        reader.round();
        let all_unfrozen = reader.ranks.iter().all(|r| {
            let st = r.scheduler.lock().peer_stats();
            st.frozen_targets == 0 && st.frozen_blocks == 0
        });
        if all_unfrozen {
            break;
        }
        assert!(
            Instant::now() < unfreeze_deadline,
            "dead-holder landing must unfreeze after retry budget + quarantine"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    reader.assert_no_leaked_pins();
}

#[test]
fn holder_without_disk_push_truncates_to_device_host_instead_of_missing() {
    let semantic = [0x9D; 32];
    let arena = Arena::new(ARENA_BASE_OFFSET as usize + 16 * 0x80_000);
    // Holder has Disk locally, but does NOT advertise Disk push (no staging ring).
    let mut holder = MultiRankReplica::start_with_options(
        8,
        40,
        HOLDER_DEVICE_LCM,
        true,
        false,
        true,
        ARENA_BASE_OFFSET,
        &arena,
        semantic,
        10_000_000,
        10_000,
        None,
    );
    let mut reader = MultiRankReplica::start(
        8,
        41,
        READER_DEVICE_LCM,
        false,
        true,
        ARENA_BASE_OFFSET + 8 * 0x80_000,
        &arena,
        semantic,
    );

    let base_prefix: Vec<i32> = (1..=8 * PAGE).collect();
    let mut ext_prefix = base_prefix.clone();
    ext_prefix.extend(60_000..60_000 + 4 * PAGE);
    holder.run_to_finish("trunc-ext", ext_prefix.clone());
    holder.evict_all_device_and_host(45_000, 8, 12);
    // Re-warm only the 8-page prefix on Device while pages 8..12 remain on Disk.
    holder.run_to_finish("trunc-base-hot", base_prefix.clone());
    holder.drain_disk_cleanings();

    let mut prompt = ext_prefix.clone();
    prompt.extend(99_000..99_000 + PAGE);
    let (landed_prefix, _) =
        drive_peer_landing(&mut holder, &mut reader, "tp8-trunc-dev-hit", prompt);
    assert_eq!(
        landed_prefix,
        base_prefix.len() as i32,
        "holder without Disk push must truncate cleanly to its Device/Host prefix instead of missing"
    );
    assert!(
        reader
            .last_offered_pages
            .iter()
            .all(|p| p.tier != PeerTier::Disk),
        "truncated offer must contain no Disk pages: {:?}",
        reader.last_offered_pages
    );
    reader.finish_request("tp8-trunc-dev-hit");
    holder.assert_no_leaked_pins();
    reader.assert_no_leaked_pins();
}

#[test]
fn tp8_kvp8_mixed_device_and_disk_and_clean_abort_release() {
    let semantic = [0x6E; 32];
    let arena = Arena::new(ARENA_BASE_OFFSET as usize + 16 * 0x80_000);
    let mut holder = MultiRankReplica::start(
        8,
        30,
        HOLDER_DEVICE_LCM,
        true,
        true,
        ARENA_BASE_OFFSET,
        &arena,
        semantic,
    );
    let mut reader = MultiRankReplica::start(
        8,
        31,
        READER_DEVICE_LCM,
        false,
        true,
        ARENA_BASE_OFFSET + 8 * 0x80_000,
        &arena,
        semantic,
    );

    // Warm a 12-page prompt (`base_prefix` + 4 tail pages) and push ALL of it onto Disk.
    let base_prefix: Vec<i32> = (1..=8 * PAGE).collect();
    let mut ext_prefix = base_prefix.clone();
    ext_prefix.extend(50_000..50_000 + 4 * PAGE);
    holder.run_to_finish("mix-ext", ext_prefix.clone());
    holder.evict_all_device_and_host(40_000, 8, 12);

    // Re-warm only the 8-page prefix on holder so slots 0..8 live on Device while
    // slots 8..12 and the 12-page State checkpoint remain exclusively on Disk!
    holder.run_to_finish("mix-base-hot", base_prefix.clone());
    holder.drain_disk_cleanings();

    // Write deterministic non-zero bytes into holder's Device/Host arena for
    // slots 0..8 read via one-sided READs concurrently with Disk pushes for slots 8..12.
    for rank in &holder.ranks {
        let rank_kv = rank.kv_dev_base;
        let rank_st = rank.st_dev_base;
        for blk in 1..HOLDER_DEVICE_LCM as u64 {
            arena.write(rank_kv + blk * BLOCK_BYTES as u64, &vec![0x71; BLOCK_BYTES]);
            arena.write(
                rank_st + blk * STATE_HEAD_BYTES as u64,
                &vec![0x72; STATE_HEAD_BYTES],
            );
        }
        let host_base = rank_kv + 0x60_000;
        for blk in 0..HOST_LCM_BLOCKS as u64 {
            arena.write(
                host_base + blk * BLOCK_BYTES as u64,
                &vec![0x73; BLOCK_BYTES],
            );
        }
    }

    let mut prompt = ext_prefix.clone();
    prompt.extend(95_000..95_000 + PAGE);
    let (landed_prefix, fetch) =
        drive_peer_landing(&mut holder, &mut reader, "tp8-mixed-hit", prompt);
    assert_eq!(landed_prefix, ext_prefix.len() as i32);
    let offered = reader.last_offered_pages.clone();
    assert!(
        offered
            .iter()
            .any(|p| matches!(p.tier, PeerTier::Device | PeerTier::Host)),
        "expected Device/Host pages in mixed offer: {offered:?}"
    );
    assert!(
        offered.iter().any(|p| p.tier == PeerTier::Disk),
        "expected Disk pages in mixed offer: {offered:?}"
    );
    verify_landed_bytes(&holder, &reader, &arena, &offered, &fetch);

    reader.finish_request("tp8-mixed-hit");
    holder.assert_no_leaked_pins();
    reader.assert_no_leaked_pins();

    // Clean abort: a reader with push_us = 0 cleanly returns PushOutcome::Aborted
    // before sending any push -> ReadRetirement::Aborted -> Release(Abort) ->
    // holder unpins cleanly, reader falls back with 0 reused tokens and no FetchFrozen.
    let mut clean_reader = MultiRankReplica::start_with_push_us(
        8,
        32,
        READER_DEVICE_LCM,
        false,
        true,
        ARENA_BASE_OFFSET + 8 * 0x80_000,
        &arena,
        semantic,
        0,
    );
    let mut abort_prompt = ext_prefix.clone();
    abort_prompt.extend(98_000..98_000 + PAGE);
    let (abort_prefix, _) = drive_peer_landing(
        &mut holder,
        &mut clean_reader,
        "tp8-clean-abort",
        abort_prompt,
    );
    assert_eq!(
        abort_prefix, 0,
        "clean abort must fall back with 0 reused tokens"
    );
    assert!(
        !clean_reader.saw_frozen,
        "clean abort must not emit FetchFrozen"
    );
    clean_reader.finish_request("tp8-clean-abort");
    holder.assert_no_leaked_pins();
    clean_reader.assert_no_leaked_pins();
}
