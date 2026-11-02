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

//! The embedded scheduler against the embedded interpreter: Python builds a
//! scheduler the way the data plane does, Rust drives it, and a plan crosses
//! back to Python.

use std::{ffi::CString, sync::Once};

use pyo3::{prelude::*, types::PyDict};
use ts_scheduler::{
    BlockTableView, CacheEvent, CacheEventKind, CacheOpKind, ForwardEvent, ForwardEventKind,
    RequestSpec, Scheduler, StorageCancellationView, StorageDoneView, StorageOpKind, StorageRound,
    StorageTickView, TransferBatchView,
};

/// Start the interpreter once per test process, with the scheduler module
/// built in and the repository's `tokenspeed_scheduler` package importable.
fn interpreter() {
    static START: Once = Once::new();
    START.call_once(|| {
        // The test binary is not in a Python prefix; point at the one the
        // crate was built against.
        std::env::set_var("PYTHONHOME", env!("TS_SCHEDULER_TEST_PYTHONHOME"));
        ts_scheduler::register_builtin_module();
        Python::initialize();
        Python::attach(|py| {
            ts_scheduler::alias_builtin_module(py).unwrap();
            let package_dir = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../tokenspeed-scheduler/python"
            );
            py.import("sys")
                .unwrap()
                .getattr("path")
                .unwrap()
                .call_method1("insert", (0, package_dir))
                .unwrap();
        });
    });
}

/// Run Python `code` and return its global `result`.
fn run<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyAny> {
    let globals = PyDict::new(py);
    py.run(&CString::new(code).unwrap(), Some(&globals), None)
        .unwrap_or_else(|e| panic!("{e}\n{code}"));
    globals
        .get_item("result")
        .unwrap()
        .expect("code sets `result`")
}

const NEW_SCHEDULER: &str = r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.prefix_granularity = 16
cfg.max_scheduled_tokens = 512
cfg.max_batch_size = 8
cfg.num_device_pages = 64
cfg.num_snapshot_pages = 1
cfg.cache_groups = [
    ts.CacheGroupConfig(
        group_id="full_attention",
        block_granularity=16,
        total_pages=64,
        attn_kind=ts.AttnKind.Full,
    )
]
result = ts.Scheduler(cfg)
"#;

fn spec(request_id: &str, tokens: Vec<i32>) -> RequestSpec {
    RequestSpec {
        request_id: request_id.to_owned(),
        tokens,
        max_new_tokens: 4,
        max_cached_prefix_tokens: i32::MAX,
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

#[test]
fn the_package_imports_the_built_in_module() {
    interpreter();
    Python::attach(|py| {
        let origin = run(
            py,
            "import tokenspeed_scheduler.tokenspeed_scheduler_ext as m\n\
             import importlib.machinery\n\
             result = (m.__spec__.origin, type(m.__spec__.loader).__name__, m.__name__)",
        );
        let (origin, loader, name): (String, String, String) = origin.extract().unwrap();
        assert_eq!(
            (origin.as_str(), loader.as_str()),
            ("built-in", "type"),
            "{name}"
        );
    });
}

#[test]
fn a_request_runs_from_prefill_to_finish() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());

    scheduler.submit_requests(&[spec("r1", (1..=10).collect())]);
    assert_eq!(scheduler.sizes().waiting, 1);

    let plan = scheduler.next_execution_plan();
    let batches = plan.forward_batches();
    assert_eq!(batches.len(), 1);
    let prefill = &batches[0];
    assert_eq!(prefill.request_ids, ["r1"]);
    assert_eq!(
        (prefill.num_extends, prefill.input_lengths.clone()),
        (1, vec![10])
    );
    assert_eq!(prefill.extend_prefix_lens, [0]);
    assert!(plan.aborts().is_empty());

    scheduler
        .advance(&[event(ForwardEventKind::ExtendResult, "r1", vec![42])])
        .unwrap();
    let decode = scheduler.next_execution_plan();
    let batches = decode.forward_batches();
    assert_eq!(
        (batches[0].num_extends, batches[0].input_lengths.clone()),
        (0, vec![1])
    );
    assert_eq!(scheduler.sizes().decoding, 1);

    // The data plane reads the same plan the control thread saw.
    Python::attach(|py| {
        let plan = decode.into_python(py).unwrap();
        let globals = PyDict::new(py);
        globals.set_item("plan", plan).unwrap();
        let code = c"batch = plan.forward[0]\n\
result = (list(batch.request_ids), list(batch.input_lengths), list(batch.request_pool_indices))";
        py.run(code, Some(&globals), None).unwrap();
        let seen: (Vec<String>, Vec<i32>, Vec<i32>) = globals
            .get_item("result")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(
            seen,
            (
                batches[0].request_ids.clone(),
                batches[0].input_lengths.clone(),
                batches[0].request_pool_indices.clone()
            )
        );
    });

    scheduler
        .advance(&[
            event(ForwardEventKind::ExtendResult, "r1", vec![43]),
            event(ForwardEventKind::Finish, "r1", Vec::new()),
        ])
        .unwrap();
    // Finishing settles once the next plan is built.
    scheduler.next_execution_plan();
    let sizes = scheduler.sizes();
    assert_eq!((sizes.waiting, sizes.prefilling, sizes.decoding), (0, 0, 0));
}

#[test]
fn an_event_the_state_does_not_allow_is_an_error() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());
    scheduler.submit_requests(&[spec("r0", vec![1, 2, 3, 4])]);
    let error = scheduler
        .advance(&[event(ForwardEventKind::Finish, "r0", Vec::new())])
        .unwrap_err();
    assert!(
        error.what().contains("FSM transition invalid"),
        "{}",
        error.what()
    );
}

#[test]
fn only_a_scheduler_object_is_accepted() {
    interpreter();
    Python::attach(|py| {
        let not_a_scheduler = run(py, "result = object()");
        let error = Scheduler::from_python(&not_a_scheduler).err().unwrap();
        assert!(error.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
    });
}

#[test]
fn a_forced_retraction_stores_and_restores_through_cache_events() {
    interpreter();
    let code = NEW_SCHEDULER.replace(
        "result = ts.Scheduler(cfg)",
        "cfg.num_snapshot_pages = 9\n\
         cfg.host_group_packing = [1]\n\
         cfg.max_retracted_requests = 4\n\
         cfg.debug_force_retraction_interval = 1\n\
         result = ts.Scheduler(cfg)",
    );
    let mut scheduler = Python::attach(|py| Scheduler::from_python(&run(py, &code)).unwrap());
    scheduler.submit_requests(&[spec("r1", (1..=10).collect())]);

    // Decode until the scheduler retracts the request: its plan stores the
    // image and runs no forward for it.
    let mut store = None;
    for token in 42..50 {
        let plan = scheduler.next_execution_plan();
        let ops = plan.cache_ops();
        if let Some(op) = ops.first() {
            assert_eq!(ops.len(), 1);
            assert_eq!(op.kind, CacheOpKind::SnapshotStore);
            assert!(plan
                .forward_batches()
                .iter()
                .all(|b| b.request_ids.is_empty()));
            store = Some(op.op_id);
            break;
        }
        scheduler
            .advance(&[event(ForwardEventKind::ExtendResult, "r1", vec![token])])
            .unwrap();
    }
    let store = store.expect("the forced retraction never came");
    scheduler
        .advance_head(
            &StorageRound::default(),
            &[CacheEvent {
                kind: CacheEventKind::SnapshotDone,
                op_id: store,
            }],
        )
        .unwrap();

    // The restore comes back as its own op; once acknowledged, the request
    // is decoding again.
    let mut restored = false;
    for _ in 0..4 {
        let plan = scheduler.next_execution_plan();
        let ops = plan.cache_ops();
        if let Some(op) = ops
            .iter()
            .find(|op| op.kind == CacheOpKind::SnapshotRestore)
        {
            scheduler
                .advance_head(
                    &StorageRound::default(),
                    &[CacheEvent {
                        kind: CacheEventKind::RestoreDone,
                        op_id: op.op_id,
                    }],
                )
                .unwrap();
            restored = true;
            break;
        }
    }
    assert!(restored, "the retracted request was never restored");
    let sizes = scheduler.sizes();
    assert_eq!((sizes.decoding, sizes.retracted), (1, 0));
}

/// A small scheduler with the L3 tier and its storage credits.
const STORAGE_SCHEDULER: &str = r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.Fused
cfg.prefix_granularity = 2
cfg.num_device_pages = 32
cfg.num_host_pages = 8
cfg.host_group_packing = [1]
cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteThrough
cfg.num_snapshot_pages = 1
cfg.max_scheduled_tokens = 8
cfg.max_batch_size = 4
cfg.disable_l2_cache = False
cfg.enable_l3_storage = True
cfg.storage_config = ts.StorageConfig(
    8, 8, 4, 4, 32, 1, 1000, 1000,
    ts.StoragePressure(8, 65536, 1000),
    ts.StoragePressure(4, 32768, 500),
)
cfg.cache_groups = [ts.CacheGroupConfig("full", 2, 32, ts.AttnKind.Full)]
result = ts.Scheduler(cfg)
"#;

/// A round head that allows forwards and storage at `now_us`.
fn tick(now_us: i64) -> StorageRound {
    StorageRound {
        tick: Some(StorageTickView {
            now_us,
            allow_forward: true,
            allow_storage: true,
            queued_operations: 0,
            queued_bytes: 0,
            oldest_operation_us: 0,
        }),
        ..StorageRound::default()
    }
}

#[test]
fn storage_operations_stay_out_of_the_device_cache_ops() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, STORAGE_SCHEDULER)).unwrap());
    scheduler.advance_head(&tick(1), &[]).unwrap();
    scheduler.submit_requests(&[spec("waiting", vec![1, 2, 3, 4])]);

    // The prompt's probe runs on the storage executors: the device's cache
    // ops never count it, and the request waits for its answer.
    let plan = scheduler.next_execution_plan();
    assert!(plan.cache_ops().is_empty());
    assert!(!plan.needs_device());
    let ops = plan.storage_ops();
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].kind, StorageOpKind::Query);
    // The prompt's two prefix pages, in order; a query names no Host block.
    assert_eq!(ops[0].keys.len(), 2);
    assert!(ops[0].keys.iter().all(|key| key.group_id == 0));
    assert_ne!(ops[0].keys[0].content_hash, ops[0].keys[1].content_hash);
    assert!(ops[0].host_blocks.is_empty());
    assert!(plan
        .forward_batches()
        .iter()
        .all(|b| b.request_ids.is_empty()));
    assert!(scheduler.storage_pending());

    // No owner holds a key: the request admits without a stored prefix.
    let keys = ops[0].keys.len();
    let done = StorageRound {
        done: vec![StorageDoneView {
            op_id: ops[0].op_id,
            success: vec![0; keys],
            retired: true,
            object_failures: vec![0; keys],
        }],
        ..tick(2)
    };
    scheduler.advance_head(&done, &[]).unwrap();
    let plan = scheduler.next_execution_plan();
    assert!(plan
        .forward_batches()
        .iter()
        .any(|b| b.request_ids == ["waiting"]));
}

#[test]
fn a_detached_request_cancels_its_storage_operation() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, STORAGE_SCHEDULER)).unwrap());
    scheduler.advance_head(&tick(1), &[]).unwrap();
    scheduler.submit_requests(&[spec("waiting", vec![1, 2, 3, 4])]);
    let query = scheduler.next_execution_plan().storage_ops()[0].clone();

    let detach = StorageRound {
        cancellations: vec![StorageCancellationView {
            request_id: "waiting".to_owned(),
            abort: false,
        }],
        ..tick(2)
    };
    scheduler.advance_head(&detach, &[]).unwrap();
    scheduler.submit_requests(&[spec("next", vec![5, 6, 7, 8])]);
    // The cancel goes out before the next request's probe.
    let ops = scheduler.next_execution_plan().storage_ops();
    assert_eq!(
        ops.iter().map(|op| op.kind).collect::<Vec<_>>(),
        [StorageOpKind::Cancel, StorageOpKind::Query]
    );
    assert_eq!(ops[0].op_id, query.op_id);
    assert!(ops[1].op_id > query.op_id);
}

/// A small scheduler of `role` whose one group transfers its suffix.
fn pd_scheduler(role: &str) -> String {
    NEW_SCHEDULER.replace(
        "result = ts.Scheduler(cfg)",
        &format!(
            "cfg.role = ts.SchedulerConfig.Role.{role}\n\
             cfg.cache_groups[0].transfer_policy = ts.CacheTransferPolicy.FullSuffix\n\
             result = ts.Scheduler(cfg)"
        ),
    )
}

/// What the data plane reads off a plan's batch `stream`: its block tables
/// (padded with -1, as `block_tables_arrays` views them) and transfer
/// columns, in the shape of [`TransferBatchView`].
fn python_transfer_view(plan: ts_scheduler::ExecutionPlan, stream: &str) -> TransferBatchView {
    Python::attach(|py| {
        let globals = PyDict::new(py);
        globals
            .set_item("plan", plan.into_python(py).unwrap())
            .unwrap();
        let code = format!(
            "batch = plan.{stream}\n\
             tables = batch.block_tables\n\
             candidates = [list(ids) for ids in batch.spec_candidate_ids]\n\
             offsets = [0]\n\
             for ids in candidates:\n    offsets.append(offsets[-1] + len(ids))\n\
             result = (list(batch.request_ids), list(batch.request_pool_indices),\n\
                       list(batch.input_lengths), list(batch.prefill_lengths),\n\
                       list(batch.extend_prefix_lens), [bool(c) for c in batch.state_prefix_cached],\n\
                       list(batch.decode_input_ids), [i for ids in candidates for i in ids], offsets,\n\
                       batch.num_extends(),\n\
                       [(g, len(rows[0]) if rows else 0, [b for row in rows for b in row])\n\
                        for g, rows in sorted(tables.items())])"
        );
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        #[allow(clippy::type_complexity)]
        let (
            request_ids,
            request_pool_indices,
            input_lengths,
            prefill_lengths,
            extend_prefix_lens,
            state_prefix_cached,
            decode_input_ids,
            spec_candidate_ids,
            spec_candidate_offsets,
            num_extends,
            tables,
        ): (
            Vec<String>,
            Vec<i32>,
            Vec<i32>,
            Vec<i32>,
            Vec<i32>,
            Vec<bool>,
            Vec<i32>,
            Vec<i32>,
            Vec<usize>,
            usize,
            Vec<(String, usize, Vec<i32>)>,
        ) = globals
            .get_item("result")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        TransferBatchView {
            request_ids,
            request_pool_indices,
            input_lengths,
            prefill_lengths,
            extend_prefix_lens,
            state_prefix_cached,
            decode_input_ids,
            spec_candidate_ids,
            spec_candidate_offsets,
            num_extends,
            block_tables: tables
                .into_iter()
                .map(|(group_id, columns, blocks)| BlockTableView {
                    group_id,
                    columns,
                    blocks,
                })
                .collect(),
        }
    })
}

#[test]
fn a_remote_prefill_carries_what_the_data_plane_reads() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, &pd_scheduler("D"))).unwrap());
    scheduler.submit_requests(&[spec("r1", (1..=40).collect())]);
    scheduler.next_execution_plan();
    scheduler
        .advance(&[event(ForwardEventKind::PdBootstrapped, "r1", Vec::new())])
        .unwrap();
    let plan = scheduler.next_execution_plan();
    let rows = plan
        .remote_prefill_transfer()
        .expect("the admission's stream");
    assert_eq!(rows.request_ids, ["r1"]);
    assert_eq!(
        (rows.num_extends, rows.prefill_lengths.clone()),
        (1, vec![40])
    );
    let [table] = &rows.block_tables[..] else {
        panic!("one group's table: {:?}", rows.block_tables);
    };
    assert_eq!(table.group_id, "full_attention");
    // 40 tokens over 16-token blocks: three real blocks.
    assert_eq!(table.blocks.len(), table.columns);
    assert!(table.blocks[..3].iter().all(|&block| block > 0));
    assert!(plan.remote_decode_transfer().is_none());
    assert_eq!(python_transfer_view(plan, "remote_prefill"), rows);
}

#[test]
fn a_remote_decode_carries_the_first_token_and_the_prompt_blocks() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, &pd_scheduler("P"))).unwrap());
    scheduler.submit_requests(&[spec("r1", (1..=20).collect())]);
    scheduler.next_execution_plan();
    scheduler
        .advance(&[event(ForwardEventKind::PdBootstrapped, "r1", Vec::new())])
        .unwrap();
    let prefill = scheduler.next_execution_plan();
    let forward = prefill.forward_transfer().expect("the prompt's batch");
    assert_eq!(
        (forward.num_extends, forward.input_lengths.clone()),
        (1, vec![20])
    );
    assert_eq!(python_transfer_view(prefill, "forward[0]"), forward);
    scheduler
        .advance(&[ForwardEvent {
            spec_candidate_ids: vec![42, 7, 8],
            ..event(ForwardEventKind::ExtendResult, "r1", vec![42])
        }])
        .unwrap();
    let plan = scheduler.next_execution_plan();
    let rows = plan
        .remote_decode_transfer()
        .expect("the prompt's KV goes out");
    assert_eq!(rows.request_ids, ["r1"]);
    assert_eq!(rows.num_extends, 0);
    assert_eq!(rows.decode_input_ids, [42]);
    assert_eq!(
        (
            rows.spec_candidate_ids.clone(),
            rows.spec_candidate_offsets.clone()
        ),
        (vec![42, 7, 8], vec![0, 3])
    );
    assert_eq!(rows.block_tables[0].blocks, forward.block_tables[0].blocks);
    assert_eq!(python_transfer_view(plan, "remote_decode"), rows);
}
