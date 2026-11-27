//! End-to-end multi-rank PyO3 integration tests for combining Disk-tier peer
//! prefix sharing with DFLASH head-sharded sliding-window draft caches:
//! 3-group layout (`draft` SWA with sink + window, `full` FullSuffix
//! KVP-sharded, `state` LatestSnapshot head-sharded).

use std::{
    collections::HashMap,
    ffi::CString,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
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
        push::{PagePusher, PageWriter, PushEndpoint, PushSource},
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

// Prefix page = 4 tokens.
// Group 0 ("draft"): sliding_window, block_granularity = 2 (2 draft blocks per prefix page),
//   sliding_window = 8 (4 draft blocks), sink_tokens = 2 (slot 0), shard_count = 1,
//   head-partitioned on axis=1 across 8 KV heads (shape [2, heads, 64]).
//   Sizing: on TP1 (width=1), 1 draft block = 2 * 8 * 64 = 1024 bytes, packing = 4 -> 4096 B/parent.
//   On TP8 (width=8), 1 draft block = 2 * 1 * 64 = 128 bytes, packing = 4 -> 512 B/parent (padded to 4096 B LCM).
// Group 1 ("full"): full_history, block_granularity = 4, shard_count = width, 4096 B/block.
// Group 2 ("state"): state, shard_count = 1, head-partitioned on axis=0 across 8 heads (512 B/head).
const PAGE: i32 = 4;
const DRAFT_BLOCK_TOKENS: i32 = 2;
const DRAFT_WINDOW_TOKENS: i32 = 8;
const DRAFT_SINK_TOKENS: i32 = 2;
const DRAFT_GLOBAL_HEADS: usize = 8;
const DRAFT_HEAD_BYTES: usize = 64; // per token per head
const BLOCK_BYTES: usize = 4096;
const STATE_GLOBAL_HEADS: usize = 8;
const STATE_HEAD_BYTES: usize = 512; // 8 * 512 == 4096
const HOLDER_DEVICE_LCM: usize = 12;
const READER_DEVICE_LCM: usize = 48;
const HOST_LCM_BLOCKS: usize = 12;
const DISK_LCM_BLOCKS: usize = 128;
const ARENA_BASE_OFFSET: u64 = 0x10_000;

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
            let sys = py.import("sys").unwrap();
            let meta_path = sys.getattr("meta_path").unwrap();
            let filtered: Vec<Py<PyAny>> = meta_path
                .try_iter()
                .unwrap()
                .filter_map(|item| {
                    let obj = item.ok()?;
                    let repr = obj.repr().ok()?.to_str().ok()?.to_owned();
                    if repr.contains("tokenspeed_scheduler") || repr.contains("_editable_skbc") {
                        None
                    } else {
                        Some(obj.unbind())
                    }
                })
                .collect();
            sys.setattr("meta_path", filtered).unwrap();
            sys.getattr("path")
                .unwrap()
                .call_method1("insert", (0, package))
                .unwrap();
        });
    });
}

fn draft_block_bytes(width: usize) -> usize {
    (DRAFT_BLOCK_TOKENS as usize) * (DRAFT_GLOBAL_HEADS / width) * DRAFT_HEAD_BYTES
}

fn dflash_contract_wire(width: usize, device_lcm_pages: usize) -> Vec<u8> {
    let draft_heads = (DRAFT_GLOBAL_HEADS / width) as i64;
    let draft_bytes = draft_block_bytes(width) as i64;
    // Draft packs 2 blocks per LCM parent (matching 2 draft blocks per prefix page).
    let draft_packing: i64 = 2;
    let draft_page_count = 1 + ((device_lcm_pages - 1) as i64) * draft_packing;
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
                    "group_id": "draft",
                    "cache_blocks_per_lcm_block": draft_packing,
                    "page_count": draft_page_count
                },
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
                { "plane_id": "dr", "bytes_per_lcm_block": draft_packing * draft_bytes, "arena_offset_bytes": 0 },
                { "plane_id": "kv", "bytes_per_lcm_block": lcm, "arena_offset_bytes": num_lcm * draft_packing * draft_bytes },
                { "plane_id": "st", "bytes_per_lcm_block": state_bytes, "arena_offset_bytes": num_lcm * (draft_packing * draft_bytes + lcm) }
            ],
            "fields": [
                {
                    "group_id": "draft",
                    "field_id": "layer.48.k",
                    "plane_id": "dr",
                    "shape": [DRAFT_BLOCK_TOKENS as i64, draft_heads, DRAFT_HEAD_BYTES as i64],
                    "dtype": "uint8",
                    "field_offset_bytes": 0,
                    "page_stride_bytes": draft_bytes
                },
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
                "group_id": "draft",
                "attn_kind": "sliding_window",
                "rows_per_page": DRAFT_BLOCK_TOKENS as i64,
                "entry_stride_tokens": 1,
                "sliding_window_tokens": DRAFT_WINDOW_TOKENS as i64,
                "sink_tokens": DRAFT_SINK_TOKENS as i64,
                "replayable": false,
                "transfer_policy": "full_suffix",
                "shard_count": 1
            },
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
                    "field_id": "layer.48.k",
                    "partition": {
                        "axis": 1,
                        "global_extent": DRAFT_GLOBAL_HEADS as i64,
                        "global_parts": []
                    }
                },
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

#[allow(clippy::too_many_arguments)]
fn make_scheduler(
    name: &str,
    incarnation: u64,
    width: usize,
    device_lcm_pages: usize,
    with_disk: bool,
    versioned_shards: bool,
    min_clean_tokens: i32,
    max_write_host_blocks: usize,
) -> Scheduler {
    interpreter();
    let draft_packing = 2usize;
    let draft_total_pages = 1 + (device_lcm_pages - 1) * draft_packing;
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
             cfg.host_group_packing = [{draft_packing}, {width}, 1]\n\
             cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack\n\
             cfg.num_disk_pages = {disk_pages}\n\
             cfg.disk_config = ts.DiskConfig(max_operations=4, max_blocks_per_operation=64,\n\
                                             max_host_blocks={max_host_blocks}, max_write_host_blocks={max_write_host_blocks},\n\
                                             max_queued_cleanings=256, min_prefetch_tokens=1,\n\
                                             wait_timeout_us=10_000_000, ready_timeout_us=10_000_000,\n\
                                             min_clean_tokens={min_clean_tokens}, require_reuse_for_cleaning=False)",
            host_pages = HOST_LCM_BLOCKS + 1,
            disk_pages = DISK_LCM_BLOCKS + 1,
            max_host_blocks = (8 * width).max(max_write_host_blocks),
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
        group_id="draft",
        block_granularity={DRAFT_BLOCK_TOKENS},
        total_pages={draft_total_pages},
        attn_kind=ts.AttnKind.SlidingWindow,
        sliding_window_tokens={DRAFT_WINDOW_TOKENS},
        sink_tokens={DRAFT_SINK_TOKENS},
        cache_blocks_per_lcm_block={draft_packing},
        transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        shard_count=1,
    ),
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
    // Use 4096 B slot size for Disk direct-IO alignment; draft packs 2 blocks per LCM parent
    // inside a 8192 B LCM block so each draft block has 4096 B stride (or 2048 B within 4096 B?
    // ts_disk::ALIGNMENT == 4096! So every group's `block_bytes` for Disk I/O must be 4096-aligned!).
    // Wait: let's check! Each draft block on Disk is 4096 B (containing the field at offset 0),
    // with packing=2 inside an 8192 B LCM parent! That matches real hardware (393,216 B is 4096-aligned).
    HostLayout {
        groups: vec![
            GroupLayout {
                packing: 2,
                block_bytes: BLOCK_BYTES,
                shard_count: 1,
            },
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
        lcm_block_bytes: 2 * BLOCK_BYTES,
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

struct ArenaWriter {
    ring: Arc<Memory>,
    arena: Arena,
    total_spans_written: Arc<AtomicU64>,
}

impl PageWriter for ArenaWriter {
    fn write(&self, _session: &str, spans: &Spans) -> Result<(), String> {
        self.total_spans_written
            .fetch_add(spans.lengths.len() as u64, Ordering::Relaxed);
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

fn peer_settings(name: &str, incarnation: u64) -> PeerSettings {
    PeerSettings {
        name: name.to_owned(),
        incarnation,
        deployment_id: "dflash-exp".into(),
        weight_version: "dflash-v1".into(),
        timeouts: Timeouts {
            answer_us: 2_000_000,
            quarantine_us: 10_000,
            push_us: 5_000_000,
        },
        record_lifetime: Duration::from_secs(10),
    }
}

struct RankState {
    rank: usize,
    dr_dev_base: u64,
    kv_dev_base: u64,
    st_dev_base: u64,
    scheduler: Arc<Mutex<Scheduler>>,
    peer: Peer,
    disk: Option<(DiskTier, Memory, ScratchFile, HostLayout)>,
    acks: Vec<CacheEvent>,
    in_flight: HashMap<u64, DiskOpView>,
}

struct DflashReplica {
    name: String,
    width: usize,
    ranks: Vec<RankState>,
    on_disk: HashMap<(usize, u32, i32), Vec<u8>>,
    total_spans_written: Arc<AtomicU64>,
    next_seed: u8,
    last_offered_pages: Vec<PeerPageView>,
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

impl DflashReplica {
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
        min_clean_tokens: i32,
        max_write_host_blocks: usize,
    ) -> Self {
        let mut listener = PeerListener::bind("127.0.0.1", 0).unwrap();
        let name = format!("127.0.0.1:{}", listener.port());
        let total_spans_written = Arc::new(AtomicU64::new(0));
        let dr_bytes = draft_block_bytes(width) as u64;
        let state_bytes = ((STATE_GLOBAL_HEADS / width) * STATE_HEAD_BYTES) as u64;
        let stride_per_rank = 0x100_000u64; // 1 MiB per rank

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
            let dr_dev_base = arena_base + (rank as u64) * stride_per_rank;
            let kv_dev_base = dr_dev_base + 0x20_000;
            let st_dev_base = dr_dev_base + 0x80_000;
            let host_base = dr_dev_base + 0xA0_000;
            let region_len = stride_per_rank;

            let (disk_entry, disk_bounds) = if with_disk {
                let dir = std::env::var_os("TS_DISK_TEST_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir);
                let fid = NEXT_FILE_ID.fetch_add(1, Ordering::Relaxed);
                let file = ScratchFile(dir.join(format!(
                    "ts-dflash-disk-{}-{}-r{}.bin",
                    std::process::id(),
                    fid,
                    rank
                )));
                let lcm_bytes = 2 * BLOCK_BYTES;
                let host = Memory::map(HOST_LCM_BLOCKS * lcm_bytes).unwrap();
                let h_layout = host_layout(HOST_LCM_BLOCKS, width, rank);
                let d_layout = host_layout(DISK_LCM_BLOCKS, width, rank);
                let tier = DiskTier::start(
                    &DiskSettings {
                        file: file.0.clone(),
                        file_bytes: (DISK_LCM_BLOCKS * lcm_bytes) as u64,
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
                let bounds: Vec<i64> = disk_layout
                    .groups
                    .iter()
                    .map(|g| (1 + disk_layout.num_lcm_blocks * g.packing * g.shard_count) as i64)
                    .collect();
                push.serve(PushSource {
                    executor,
                    disk_layout,
                    writer: Arc::new(ArenaWriter {
                        ring: ring.clone(),
                        arena: arena.clone(),
                        total_spans_written: total_spans_written.clone(),
                    }),
                    ring,
                    slot_bytes,
                })
                .unwrap();
                (Some((tier, host, file, h_layout)), bounds)
            } else {
                (None, Vec::new())
            };

            let geom = RankGeometry {
                session: format!("sess-{name}-r{rank}"),
                device: vec![
                    DeviceField {
                        group_id: 0,
                        field_id: "layer.48.k".into(),
                        address: dr_dev_base,
                        page_stride: dr_bytes,
                        bytes: dr_bytes,
                    },
                    DeviceField {
                        group_id: 1,
                        field_id: "layer.0.kv".into(),
                        address: kv_dev_base,
                        page_stride: BLOCK_BYTES as u64,
                        bytes: BLOCK_BYTES as u64,
                    },
                    DeviceField {
                        group_id: 2,
                        field_id: "layer.0.st".into(),
                        address: st_dev_base,
                        page_stride: state_bytes,
                        bytes: state_bytes,
                    },
                ],
                host: Some(HostPool {
                    address: host_base,
                    groups: vec![
                        HostGroup {
                            parent_bytes: (2 * BLOCK_BYTES) as u64,
                            packing: 2,
                            block_bytes: BLOCK_BYTES as u64,
                            field_offsets: vec![("layer.48.k".into(), 0)],
                        },
                        HostGroup {
                            parent_bytes: (2 * BLOCK_BYTES) as u64,
                            packing: 1,
                            block_bytes: BLOCK_BYTES as u64,
                            field_offsets: vec![("layer.0.kv".into(), 0)],
                        },
                        HostGroup {
                            parent_bytes: (2 * BLOCK_BYTES) as u64,
                            packing: 1,
                            block_bytes: BLOCK_BYTES as u64,
                            field_offsets: vec![("layer.0.st".into(), 0)],
                        },
                    ],
                }),
                push: Some(push.name().to_owned()),
            };
            let host_bounds = vec![
                (1 + HOST_LCM_BLOCKS * 2) as i64,
                (1 + HOST_LCM_BLOCKS * width) as i64,
                (1 + HOST_LCM_BLOCKS) as i64,
            ];
            let member = Member {
                geometry: geom.clone(),
                bounds: host_bounds,
                disk_bounds,
                regions: vec![(dr_dev_base, region_len)],
            };
            member_blobs.push(member.encode(rank));
            geometries.push(geom);
            disks.push(disk_entry);
            push_eps.push(Arc::new(push));
        }

        let contract_wire = dflash_contract_wire(width, device_lcm_pages);
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
                min_clean_tokens,
                max_write_host_blocks,
            )));
            let pusher: Arc<dyn PagePusher> = push;
            let mut peer = Peer::start(
                peer_settings(&name, incarnation),
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
            )
            .unwrap();
            let dr_dev_base = arena_base + (rank as u64) * stride_per_rank;
            let kv_dev_base = dr_dev_base + 0x20_000;
            let st_dev_base = dr_dev_base + 0x80_000;
            ranks.push(RankState {
                rank,
                dr_dev_base,
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
            total_spans_written,
            next_seed: 1,
            last_offered_pages: Vec::new(),
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

    fn round(&mut self) -> (Vec<(String, i32)>, Vec<PeerFetchView>) {
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
                                if group == 0 && self.width == 1 {
                                    // TP1 Draft SWA block: shape [2 tokens, 8 heads, 64 bytes].
                                    // Fill each (token, head) distinctly to verify 2D strided slicing!
                                    for tok in 0..(DRAFT_BLOCK_TOKENS as usize) {
                                        for head in 0..DRAFT_GLOBAL_HEADS {
                                            let lo = (tok * DRAFT_GLOBAL_HEADS + head)
                                                * DRAFT_HEAD_BYTES;
                                            let fill = seed_base
                                                .wrapping_add(dblk as u8)
                                                .wrapping_add((tok as u8 + 1) * 53)
                                                .wrapping_add((head as u8 + 1) * 17)
                                                | 1;
                                            for (k, b) in page[lo..lo + DRAFT_HEAD_BYTES]
                                                .iter_mut()
                                                .enumerate()
                                            {
                                                *b = fill.wrapping_add((k & 0x3f) as u8);
                                            }
                                        }
                                    }
                                } else if group == 2 && self.width == 1 {
                                    // TP1 State block: 8 distinct 512-byte head slices.
                                    for head in 0..STATE_GLOBAL_HEADS {
                                        let lo = head * STATE_HEAD_BYTES;
                                        let fill = seed_base
                                            .wrapping_add(dblk as u8)
                                            .wrapping_add((head as u8 + 1) * 19)
                                            | 1;
                                        for (k, b) in
                                            page[lo..lo + STATE_HEAD_BYTES].iter_mut().enumerate()
                                        {
                                            *b = fill.wrapping_add((k & 0x3f) as u8);
                                        }
                                    }
                                } else {
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
            assert!(Instant::now() < deadline, "{request_id} did not prefill");
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
    holder: &mut DflashReplica,
    reader: &mut DflashReplica,
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

fn verify_dflash_landed_bytes(
    holder: &DflashReplica,
    reader: &DflashReplica,
    arena: &Arena,
    offered: &[PeerPageView],
    fetch: &PeerFetchView,
) -> (usize, usize, usize) {
    let by_slot: HashMap<(u32, i32), PeerPageView> =
        offered.iter().map(|p| ((p.group_id, p.slot), *p)).collect();
    let dr_dst_bytes = draft_block_bytes(reader.width);
    let mut draft_disk_checked = 0usize;
    let mut full_disk_checked = 0usize;
    let mut state_disk_checked = 0usize;

    for dst in &fetch.destinations {
        let src = by_slot[&(dst.group_id, dst.slot)];
        if src.tier != PeerTier::Disk {
            continue;
        }
        match dst.group_id {
            0 => {
                // Draft SWA: head-sharded on axis=1 across all 8 reader ranks!
                draft_disk_checked += 1;
                for reader_rank in 0..8 {
                    let dst_addr = reader.ranks[reader_rank].dr_dev_base
                        + (dst.block_id as u64) * (dr_dst_bytes as u64);
                    let actual = arena.read_bytes(dst_addr, dr_dst_bytes);
                    let expected = if holder.width == 1 {
                        let full_draft = &holder.on_disk[&(0, 0, src.block_id)];
                        let mut sliced = Vec::with_capacity(dr_dst_bytes);
                        for tok in 0..(DRAFT_BLOCK_TOKENS as usize) {
                            let lo = (tok * DRAFT_GLOBAL_HEADS + reader_rank) * DRAFT_HEAD_BYTES;
                            sliced.extend_from_slice(&full_draft[lo..lo + DRAFT_HEAD_BYTES]);
                        }
                        sliced
                    } else {
                        holder.on_disk[&(reader_rank, 0, src.block_id)][..dr_dst_bytes].to_vec()
                    };
                    assert_eq!(
                        actual, expected,
                        "Draft SWA Disk byte mismatch slot={} src_blk={} dst_blk={} reader_rank={}",
                        dst.slot, src.block_id, dst.block_id, reader_rank
                    );
                }
            }
            1 => {
                // Full-KV: KVP8 sharded.
                full_disk_checked += 1;
                let reader_rank = ((dst.block_id - 1) as usize) % 8;
                let local_dst = 1 + ((dst.block_id - 1) as u64) / 8;
                let holder_rank = ((src.block_id - 1) as usize) % holder.width;
                let expected = &holder.on_disk[&(holder_rank, 1, src.block_id)];
                let dst_addr =
                    reader.ranks[reader_rank].kv_dev_base + local_dst * BLOCK_BYTES as u64;
                let actual = arena.read_bytes(dst_addr, BLOCK_BYTES);
                assert_eq!(
                    actual, *expected,
                    "Full KV Disk byte mismatch slot={} src_blk={} (h={}) -> dst_blk={} (r={})",
                    dst.slot, src.block_id, holder_rank, dst.block_id, reader_rank
                );
            }
            2 => {
                // State: head-sharded on axis=0 across all 8 reader ranks.
                state_disk_checked += 1;
                for reader_rank in 0..8 {
                    let dst_addr = reader.ranks[reader_rank].st_dev_base
                        + (dst.block_id as u64) * STATE_HEAD_BYTES as u64;
                    let actual = arena.read_bytes(dst_addr, STATE_HEAD_BYTES);
                    let expected = if holder.width == 1 {
                        let full_state = &holder.on_disk[&(0, 2, src.block_id)];
                        let lo = reader_rank * STATE_HEAD_BYTES;
                        full_state[lo..lo + STATE_HEAD_BYTES].to_vec()
                    } else {
                        holder.on_disk[&(reader_rank, 2, src.block_id)][..STATE_HEAD_BYTES].to_vec()
                    };
                    assert_eq!(
                        actual, expected,
                        "State Disk byte mismatch src_blk={} dst_blk={} reader_rank={}",
                        src.block_id, dst.block_id, reader_rank
                    );
                }
            }
            other => panic!("unexpected group {other}"),
        }
    }
    (draft_disk_checked, full_disk_checked, state_disk_checked)
}

#[test]
fn tp8_dflash_disk_to_tp8_full_and_partial_prefix() {
    let semantic = [0xD1; 32];
    let arena = Arena::new(ARENA_BASE_OFFSET as usize + 16 * 0x100_000);
    let mut holder = DflashReplica::start(
        8,
        100,
        HOLDER_DEVICE_LCM,
        true,
        true,
        ARENA_BASE_OFFSET,
        &arena,
        semantic,
        0,
        16,
    );
    let mut reader = DflashReplica::start(
        8,
        101,
        READER_DEVICE_LCM,
        false,
        true,
        ARENA_BASE_OFFSET + 8 * 0x100_000,
        &arena,
        semantic,
        0,
        16,
    );

    // 6 full pages = 24 tokens -> 12 draft blocks (0..12).
    // Retained draft slots at end=24: sink {0} + window [8..12) = {0, 8, 9, 10, 11} (5 blocks!).
    let prefix: Vec<i32> = (1..=6 * PAGE).collect();
    holder.run_to_finish("dflash-seed", prefix.clone());
    holder.evict_all_device_and_host(10_000, 6, 16);

    // Case 1: Virgin reader (begin=0, end=24) pulls Draft (sink + window), Full, State from Disk!
    let mut prompt = prefix.clone();
    prompt.extend(90_000..90_000 + PAGE);
    let (landed, fetch) = drive_peer_landing(&mut holder, &mut reader, "tp8-dflash-virgin", prompt);
    assert_eq!(landed, prefix.len() as i32);
    let offered = reader.last_offered_pages.clone();
    let draft_slots: Vec<i32> = offered
        .iter()
        .filter(|p| p.group_id == 0)
        .map(|p| p.slot)
        .collect();
    assert_eq!(
        draft_slots,
        vec![0, 8, 9, 10, 11],
        "expected sink slot 0 + window slots 8..12 for DFLASH draft on Disk: {offered:?}"
    );
    let (dr_cnt, full_cnt, st_cnt) =
        verify_dflash_landed_bytes(&holder, &reader, &arena, &offered, &fetch);
    assert_eq!((dr_cnt, full_cnt, st_cnt), (5, 6, 1));
    reader.finish_request("tp8-dflash-virgin");
    holder.assert_no_leaked_pins();
    reader.assert_no_leaked_pins();

    // Case 2: Reader with partial local prefix (begin=8 > sink, end=24)!
    // Start a fresh reader, pre-warm 0..8 locally (2 full pages, draft sink 0 + window 0..4),
    // then pull [8, 24) from holder's Disk! Draft must land ONLY window {8, 9, 10, 11} (not sink 0).
    let mut reader2 = DflashReplica::start(
        8,
        102,
        READER_DEVICE_LCM,
        false,
        true,
        ARENA_BASE_OFFSET + 8 * 0x100_000,
        &arena,
        semantic,
        0,
        16,
    );
    let local_part: Vec<i32> = prefix[..(2 * PAGE) as usize].to_vec();
    reader2.run_to_finish("local-prefix-8", local_part);

    let mut prompt2 = prefix.clone();
    prompt2.extend(91_000..91_000 + PAGE);
    let (landed2, fetch2) =
        drive_peer_landing(&mut holder, &mut reader2, "tp8-dflash-partial", prompt2);
    assert_eq!(landed2, prefix.len() as i32);
    assert_eq!(fetch2.begin_tokens, 2 * PAGE);
    let dst_draft_slots: Vec<i32> = fetch2
        .destinations
        .iter()
        .filter(|p| p.group_id == 0)
        .map(|p| p.slot)
        .collect();
    assert_eq!(
        dst_draft_slots,
        vec![8, 9, 10, 11],
        "partial local prefix reader already owns sink slot 0; must land only window slots 8..12"
    );
    let (dr_cnt2, full_cnt2, st_cnt2) = verify_dflash_landed_bytes(
        &holder,
        &reader2,
        &arena,
        &reader2.last_offered_pages,
        &fetch2,
    );
    assert_eq!((dr_cnt2, full_cnt2, st_cnt2), (4, 4, 1));
    reader2.finish_request("tp8-dflash-partial");
    holder.assert_no_leaked_pins();
    reader2.assert_no_leaked_pins();
}

#[test]
fn tp1_dflash_disk_to_tp8_cross_layout_head_slicing() {
    let semantic = [0xD2; 32];
    let arena = Arena::new(ARENA_BASE_OFFSET as usize + 10 * 0x100_000);
    // TP1/KVP1 Decode holder -> TP8/KVP8 Prefill reader!
    let mut holder = DflashReplica::start(
        1,
        200,
        24,
        true,
        false,
        ARENA_BASE_OFFSET,
        &arena,
        semantic,
        0,
        16,
    );
    let mut reader = DflashReplica::start(
        8,
        201,
        READER_DEVICE_LCM,
        false,
        true,
        ARENA_BASE_OFFSET + 0x100_000,
        &arena,
        semantic,
        0,
        16,
    );

    let prefix: Vec<i32> = (1..=6 * PAGE).collect();
    holder.run_to_finish("d-dflash-seed", prefix.clone());
    // 24 Device LCM + 12 Host LCM = 36 LCM; each 6-page req uses ~13 LCM -> 6 reqs (~78 LCM)
    // fully evicts Device + Host onto Disk without overflowing 128 Disk LCM blocks!
    holder.evict_all_device_and_host(20_000, 6, 6);

    let mut prompt = prefix.clone();
    prompt.extend(80_000..80_000 + PAGE);
    let (landed, fetch) =
        drive_peer_landing(&mut holder, &mut reader, "tp1-to-tp8-dflash-disk", prompt);
    assert_eq!(landed, prefix.len() as i32);
    let offered = reader.last_offered_pages.clone();
    let (dr_cnt, full_cnt, st_cnt) =
        verify_dflash_landed_bytes(&holder, &reader, &arena, &offered, &fetch);
    assert_eq!((dr_cnt, full_cnt, st_cnt), (5, 6, 1));
    // Verify CPU gather-pack coalescing of contiguous-destination strided spans:
    // Without coalescing: 5 draft blocks * 2 rows/block * 8 reader ranks = 80 spans + 6 full + 8 state = 94.
    // With coalescing: each block's 2 strided rows pack into 1 contiguous span per rank -> 40 + 6 + 8 = 54!
    let spans = holder.total_spans_written.load(Ordering::Relaxed);
    assert_eq!(
        spans, 54,
        "expected 40 coalesced draft spans + 6 full + 8 state = 54 spans on TP1->TP8 (down from 94)"
    );
    reader.finish_request("tp1-to-tp8-dflash-disk");
    holder.assert_no_leaked_pins();
    reader.assert_no_leaked_pins();
}

#[test]
fn dflash_disk_cleaning_climbs_through_partial_on_disk_ancestor() {
    // Regression check for `DiskTransferManager::StartCleaning` (DiskTransferManager::StartCleaning):
    // Boundary node P0 has 2 Draft SWA blocks + 1 Full block = 3 dirty blocks on Host.
    // With `min_clean_tokens = 2 * PAGE` (8 tokens, so P0 at 4 tokens fails standalone admission)
    // and `max_write_host_blocks = 1` (truncating P0 mid-node after writing only its first block,
    // which flips `P0->OnDisk() == true`!), when the chain climbs again from P5, the loop
    // `above->OnHost() && !above->OnDisk()` STOPS BEFORE P0, stranding P0's remaining dirty
    // blocks (`full[0]` or `draft[0]` sink) on Host until Host eviction drops them forever!
    let semantic = [0xD3; 32];
    let arena = Arena::new(ARENA_BASE_OFFSET as usize + 16 * 0x100_000);
    let mut holder = DflashReplica::start(
        8,
        300,
        HOLDER_DEVICE_LCM,
        true,
        true,
        ARENA_BASE_OFFSET,
        &arena,
        semantic,
        2 * PAGE, // min_clean_tokens = 8 -> P0 (4 tokens) fails branch admission on its own!
        1,        // max_write_host_blocks = 1 -> truncates P0 (3 dirty blocks) after 1 block!
    );
    let mut reader = DflashReplica::start(
        8,
        301,
        READER_DEVICE_LCM,
        false,
        true,
        ARENA_BASE_OFFSET + 8 * 0x100_000,
        &arena,
        semantic,
        0,
        16,
    );

    let prefix: Vec<i32> = (1..=6 * PAGE).collect();
    holder.run_to_finish("partial-clean-seed", prefix.clone());
    holder.evict_all_device_and_host(30_000, 6, 16);

    let mut prompt = prefix.clone();
    prompt.extend(95_000..95_000 + PAGE);
    reader.submit_requests(&[req_spec("tp8-partial-clean-hit", prompt, &holder.name)]);
    let mut landed = -1;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        holder.round();
        let (rows, _) = reader.round();
        if let Some((_, p)) = rows.into_iter().find(|(r, _)| r == "tp8-partial-clean-hit") {
            landed = p;
            break;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    reader.finish_request("tp8-partial-clean-hit");
    holder.assert_no_leaked_pins();
    reader.assert_no_leaked_pins();
    // After Fix #1 (`StartCleaning` climbs through partially-Disk ancestors), P0 is fully cleaned
    // and all 24 tokens land cleanly from Disk!
    assert_eq!(
        landed, 24,
        "expected full 24 tokens landed after StartCleaning partial-node OnDisk() fix"
    );
}

#[test]
fn dflash_host_landing_tier_includes_sink_blocks_in_shared_prefix_keys() {
    // Regression test for `StorageQueries::SharedPrefixKeys`:
    // For a SlidingWindow group with `sink_tokens > 0` (DFLASH), `SharedPrefixKeys` must emit both
    // `SinkBlocksBelow` (`slot 0`) and `SlidingWindowBlocks`. Otherwise, when a reader uses
    // `peer_prefix_landing_tier = Host` (`PeerHostLandings::Start`), `ReserveExactStoragePrefix`
    // fails to match any prefix longer than `sliding_window_tokens` because sink `slot 0` is missing!
    interpreter();
    let mut sched = Python::attach(|py| {
        let code = format!(
            r#"
import tokenspeed_scheduler as ts

cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.P
cfg.prefix_granularity = {PAGE}
cfg.max_scheduled_tokens = 256
cfg.max_batch_size = 8
cfg.max_resident_requests = 8
cfg.num_device_pages = 32
cfg.num_snapshot_pages = 1
cfg.disable_l2_cache = False
cfg.num_host_pages = 32
cfg.host_group_packing = [2, 1, 1]
cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack
cfg.cache_groups = [
    ts.CacheGroupConfig(
        group_id="draft",
        block_granularity={DRAFT_BLOCK_TOKENS},
        total_pages=63,
        attn_kind=ts.AttnKind.SlidingWindow,
        sliding_window_tokens={DRAFT_WINDOW_TOKENS},
        sink_tokens={DRAFT_SINK_TOKENS},
        cache_blocks_per_lcm_block=2,
        transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        shard_count=1,
    ),
    ts.CacheGroupConfig(
        group_id="full",
        block_granularity={PAGE},
        total_pages=32,
        attn_kind=ts.AttnKind.Full,
        cache_blocks_per_lcm_block=1,
        transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        shard_count=1,
    ),
    ts.CacheGroupConfig(
        group_id="state",
        block_granularity=0,
        total_pages=32,
        attn_kind=ts.AttnKind.State,
        cache_blocks_per_lcm_block=1,
        transfer_policy=ts.CacheTransferPolicy.LatestSnapshot,
        shard_count=1,
    ),
]
cfg.peer_config = ts.PeerConfig(
    name="reader-host-landing", incarnation=500, max_leases=16,
    soft_pin_us=2_000_000, hard_pin_us=10_000_000,
    lookup_timeout_us=2_000_000, min_fetch_tokens={PAGE},
    landing_tier=ts.PeerLandingTier.Host,
    max_host_landing_blocks=32, host_ready_timeout_us=2_000_000,
    versioned_shards=False,
)
result = ts.Scheduler(cfg)
"#
        );
        let globals = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    });
    let mut tokens: Vec<i32> = (1..=24).collect();
    tokens.extend([999, 998, 997, 996]);
    sched
        .submit_requests(&[req_spec("r-host-dflash", tokens, "127.0.0.1:9999")])
        .unwrap();
    sched
        .advance(&[fwd_event(
            ForwardEventKind::PdBootstrapped,
            "r-host-dflash",
            Vec::new(),
        )])
        .unwrap();
    let _plan = sched.next_execution_plan(true, false);
    let lookups = sched.take_peer_lookups();
    assert_eq!(lookups.len(), 1);
    let lk = &lookups[0];
    sched
        .advance_peer(
            1000,
            &[ts_scheduler::PeerEventView {
                kind: PeerEventKind::LookupAnswered,
                lease: lk.lease.clone(),
                prefix_hashes: Vec::new(),
                tokens: Vec::new(),
                max_tokens: 0,
                pages: Vec::new(),
                demote: false,
                num_tokens: 24,
                dispatch_us: 0,
                writable_us: 0,
                terminal_us: 0,
                checkpoint: Vec::new(),
                request_id: String::new(),
            }],
        )
        .unwrap();
    let plan2 = sched.next_execution_plan(true, false);
    let fetch_count = plan2.peer_fetches().len();
    assert_eq!(
        fetch_count, 1,
        "expected PeerHostLandings to start 1 fetch after SharedPrefixKeys includes SinkBlocksBelow!"
    );
}
