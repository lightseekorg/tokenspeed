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

//! A replica's Disk tier over a real scheduler, a real file and a Host pool
//! in memory (`docs/rfcs/0006-disk-tier.md`): pages the Host evicts after
//! cleaning come back from the file byte for byte before their prompt is
//! admitted. The file lives under `TS_DISK_TEST_DIR`, or the temporary
//! directory, which must support `O_DIRECT` (not tmpfs).

use std::{
    collections::HashMap,
    ffi::CString,
    path::PathBuf,
    sync::Once,
    time::{Duration, Instant},
};

use pyo3::{prelude::*, types::PyDict};
use ts_control::disk::{DiskSettings, DiskTier};
use ts_disk::{Backend, ExecutorConfig, Memory};
use ts_l3::{GroupLayout, HostBuffer, HostLayout};
use ts_scheduler::{
    CacheEvent, CacheEventKind, CacheOpKind, DiskOpKind, DiskOpView, ForwardEvent,
    ForwardEventKind, RequestSpec, Scheduler,
};

const PAGE: i32 = 4;
const BLOCK_BYTES: usize = 4096;
// Eight usable Host blocks: each eviction write-back is placed whole, and
// the Host drops its oldest entries, already on Disk, to place the next.
const HOST_BLOCKS: usize = 8;
const DISK_BLOCKS: usize = 32;

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

/// A fused scheduler with eight usable Device blocks over a write-back Host
/// tier and a Disk tier.
fn scheduler() -> Scheduler {
    interpreter();
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
cfg.num_device_pages = 9
cfg.num_snapshot_pages = 9
cfg.max_retracted_requests = 8
cfg.disable_l2_cache = False
cfg.num_host_pages = {host_pages}
cfg.host_group_packing = [1]
cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteBack
cfg.num_disk_pages = {disk_pages}
cfg.disk_config = ts.DiskConfig(max_operations=4, max_blocks_per_operation=64,
                                max_host_blocks=4, max_write_host_blocks=2,
                                max_queued_cleanings=64, min_prefetch_tokens=1,
                                wait_timeout_us=10_000_000, ready_timeout_us=10_000_000)
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full", block_granularity={PAGE},
                                        total_pages=9, attn_kind=ts.AttnKind.Full)]
result = ts.Scheduler(cfg)
"#,
            host_pages = HOST_BLOCKS + 1,
            disk_pages = DISK_BLOCKS + 1,
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

struct ScratchFile(PathBuf);

impl ScratchFile {
    fn new(name: &str) -> Self {
        let dir = std::env::var_os("TS_DISK_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        Self(dir.join(format!("ts-control-disk-{}-{name}", std::process::id())))
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// One rank's replica: the scheduler, its Disk tier, and the data plane's
/// part played in memory. Device copies finish at once; each page the Host
/// receives is a byte pattern of its own.
struct Replica {
    scheduler: Scheduler,
    disk: DiskTier,
    host: Memory,
    host_layout: HostLayout,
    _file: ScratchFile,
    acks: Vec<CacheEvent>,
    in_flight: HashMap<u64, DiskOpView>,
    /// The bytes each cleaned Disk block holds.
    on_disk: HashMap<i32, Vec<u8>>,
    next_pattern: u8,
    /// Blocks prefetched and found byte for byte as cleaned.
    verified_reads: usize,
}

impl Replica {
    fn start(backend: Backend) -> Self {
        let file = ScratchFile::new(&format!("{backend:?}"));
        let host = Memory::map(HOST_BLOCKS * BLOCK_BYTES).unwrap();
        let host_layout = layout(HOST_BLOCKS);
        let disk = DiskTier::start(
            &DiskSettings {
                file: file.0.clone(),
                file_bytes: (DISK_BLOCKS * BLOCK_BYTES) as u64,
                executor: ExecutorConfig {
                    backend,
                    queue_depth: 4,
                    write_queue_depth: 2,
                    max_io_bytes: 1 << 20,
                    write_bytes_per_second: 1 << 30,
                },
                host: HostBuffer {
                    base: host.address(),
                    len: host.size(),
                },
                host_layout: host_layout.clone(),
                disk_layout: layout(DISK_BLOCKS),
                max_operations: 4,
                push_ring_bytes: 0,
            },
            None,
        )
        .unwrap();
        Self {
            scheduler: scheduler(),
            disk,
            host,
            host_layout,
            _file: file,
            acks: Vec::new(),
            in_flight: HashMap::new(),
            on_disk: HashMap::new(),
            next_pattern: 1,
            verified_reads: 0,
        }
    }

    fn host_offset(&self, group: u32, block: i32) -> usize {
        self.host_layout
            .owned_range(group, block)
            .unwrap()
            .expect("one owner")
            .offset
    }

    /// One control round: agree on what the Disk tier finished, feed it and
    /// the cache acknowledgments back, plan, and start the plan's work.
    /// Returns the plan's extend rows as `(request, cached prefix)`.
    fn round(&mut self) -> Vec<(String, i32)> {
        let report = self.disk.report();
        let storage = self.disk.agree(&[report], true, true).unwrap();
        for done in &storage.disk_done {
            assert!(done.success, "Disk operation {} failed", done.op_id);
            let op = self.in_flight.remove(&done.op_id).expect("a started op");
            for row in 0..op.group_ids.len() {
                let offset = self.host_offset(op.group_ids[row], op.host_blocks[row]);
                match op.kind {
                    // The Host copy may be dropped now; nothing reads it again
                    // for this test unless a prefetch rewrites it.
                    DiskOpKind::Write => self.host.write_at(offset, &[0; BLOCK_BYTES]),
                    DiskOpKind::Read => {
                        let mut landed = vec![0; BLOCK_BYTES];
                        self.host.read_at(offset, &mut landed);
                        assert_eq!(
                            &landed, &self.on_disk[&op.disk_blocks[row]],
                            "Disk block {} came back changed",
                            op.disk_blocks[row]
                        );
                        self.verified_reads += 1;
                    }
                    _ => unreachable!(),
                }
            }
        }
        let acks = std::mem::take(&mut self.acks);
        self.scheduler.advance_head(&storage, &acks).unwrap();
        let plan = self.scheduler.next_execution_plan(true, false);
        for op in plan.disk_ops() {
            if op.kind == DiskOpKind::Write {
                // The write-back that put this page on the Host landed before
                // its cleaning was planned.
                for row in 0..op.group_ids.len() {
                    let page = vec![self.next_pattern; BLOCK_BYTES];
                    self.next_pattern = self.next_pattern.wrapping_add(1).max(1);
                    let offset = self.host_offset(op.group_ids[row], op.host_blocks[row]);
                    self.host.write_at(offset, &page);
                    self.on_disk.insert(op.disk_blocks[row], page);
                }
            }
            self.in_flight.insert(op.op_id, op);
        }
        self.disk.submit(&plan).unwrap();
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
        plan.forward_batches()
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
            .collect()
    }

    /// Admits, prefills and finishes `request`; returns the cached prefix
    /// its prefill extended.
    fn run(&mut self, request: &str, tokens: Vec<i32>) -> i32 {
        self.scheduler
            .submit_requests(&[RequestSpec {
                admission_id: 0,
                request_id: request.to_owned(),
                session_id: String::new(),
                cache_group_id: String::new(),
                shared_prefix_tokens: Vec::new(),
                peer_holder: String::new(),
                peer_expected_tokens: 0,
                tokens,
                max_new_tokens: 1,
                max_cached_prefix_tokens: i32::MAX,
            }])
            .unwrap();
        let start = Instant::now();
        let prefix = loop {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "{request} never prefilled"
            );
            if let Some((_, prefix)) = self.round().into_iter().find(|(rid, _)| rid == request) {
                break prefix;
            }
            std::thread::sleep(Duration::from_millis(1));
        };
        let event = |kind, tokens| ForwardEvent {
            kind,
            request_id: request.to_owned(),
            tokens,
            spec_candidate_ids: Vec::new(),
            reserve_num_tokens: 0,
            bootstrap_token: -1,
        };
        self.scheduler
            .advance(&[
                event(ForwardEventKind::ExtendResult, vec![900]),
                event(ForwardEventKind::Finish, Vec::new()),
            ])
            .unwrap();
        prefix
    }

    /// Runs rounds until every queued cleaning is written.
    fn settle(&mut self) {
        let start = Instant::now();
        while !self.in_flight.is_empty()
            || !self.acks.is_empty()
            || self.scheduler.disk_stats().queued_cleanings > 0
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

fn evicted_pages_come_back_from_disk(backend: Backend) {
    let mut replica = Replica::start(backend);
    let seed: Vec<i32> = (1..=8).collect();
    assert_eq!(replica.run("seed", seed.clone()), 0);
    // Each 23-token prompt evicts what came before it from the Device into
    // the Host, which drops its oldest entries once full.
    for (index, base) in [100, 200, 300].into_iter().enumerate() {
        replica.run(&format!("pressure-{index}"), (base..base + 23).collect());
        replica.settle();
    }
    let stats = replica.scheduler.disk_stats();
    assert!(stats.written_blocks >= 3, "{stats:?}");
    assert_eq!((stats.failed_writes, stats.failed_reads), (0, 0));

    let mut hit = seed;
    hit.push(9);
    let prefix = replica.run("hit", hit);
    let stats = replica.scheduler.disk_stats();
    assert!(
        stats.read_blocks > 0,
        "the seed's pages came from Disk: {stats:?}"
    );
    assert_eq!(replica.verified_reads as u64, stats.read_blocks);
    assert_eq!(prefix, 8, "both seed pages resume the prompt");
    replica.disk.close(Duration::from_secs(5)).unwrap();
}

#[test]
fn evicted_pages_come_back_from_disk_through_io_uring() {
    evicted_pages_come_back_from_disk(Backend::IoUring);
}

#[test]
fn evicted_pages_come_back_from_disk_through_psync() {
    evicted_pages_come_back_from_disk(Backend::Psync);
}
