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
    profile, BlockTableView, CacheEvent, CacheEventKind, CacheOpKind, ForwardEvent,
    ForwardEventKind, RequestSpec, Scheduler, SharedL2AnswerView, SharedL2CheckpointView,
    SharedL2LookupView, SharedL2OpKind, SharedL2Pool, SharedL2PoolStatsView,
    StorageCancellationView, StorageDoneView, StorageKeyView, StorageOpKind, StorageRound,
    StorageTickView, TransferBatchView,
};

/// Start the interpreter once per test process, with the scheduler module
/// loaded and the repository's `tokenspeed_scheduler` package importable.
fn interpreter() {
    static START: Once = Once::new();
    START.call_once(|| {
        // The test binary is not in a Python prefix; point at the one the
        // crate was built against.
        std::env::set_var("PYTHONHOME", env!("TS_SCHEDULER_TEST_PYTHONHOME"));
        Python::initialize();
        Python::attach(|py| {
            let module = ts_scheduler::python_module(py).unwrap();
            py.import("sys")
                .unwrap()
                .getattr("modules")
                .unwrap()
                .set_item("tokenspeed_scheduler.tokenspeed_scheduler_ext", module)
                .unwrap();
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
cfg.max_resident_requests = 8
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
        session_id: String::new(),
        cache_group_id: String::new(),
        shared_prefix_tokens: Vec::new(),
        peer_holder: String::new(),
        request_id: request_id.to_owned(),
        tokens,
        max_new_tokens: 4,
        max_cached_prefix_tokens: i32::MAX,
    }
}

#[test]
fn submission_errors_return_to_rust_without_inserting_the_batch() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());
    scheduler
        .submit_requests(&[spec("active", vec![1, 2])])
        .unwrap();

    let mut negative_budget = spec("invalid", vec![3]);
    negative_budget.max_new_tokens = -1;
    let mut negative_prefix = spec("invalid", vec![3]);
    negative_prefix.max_cached_prefix_tokens = -1;
    for (invalid, message) in [
        (spec("active", vec![3]), "duplicate request id"),
        (spec("fresh", vec![3]), "duplicate request id"),
        (spec("invalid", Vec::new()), "tokens must be non-empty"),
        (negative_budget, "max_new_tokens must be non-negative"),
        (
            negative_prefix,
            "max_cached_prefix_tokens must be non-negative",
        ),
        (spec("invalid", vec![3; 2000]), "exceeds cache capacity"),
    ] {
        let error = scheduler
            .submit_requests(&[spec("fresh", vec![4, 5, 6]), invalid])
            .unwrap_err();
        assert!(error.what().contains(message), "{error}");
        assert_eq!(scheduler.sizes().waiting, 1);
        assert_eq!(scheduler.request_token_size("active"), 2);
        assert_eq!(scheduler.request_token_size("fresh"), -1);
        assert_eq!(scheduler.request_token_size("invalid"), -1);
    }

    scheduler
        .submit_requests(&[spec("fresh", vec![4, 5, 6])])
        .unwrap();
    let batches = scheduler.next_execution_plan(true, false).forward_batches();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].request_ids, ["active", "fresh"]);
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
fn delayed_prefill_drains_its_reservation_and_holds_new_admissions() {
    interpreter();
    let code = NEW_SCHEDULER
        .replace("prefix_granularity = 16", "prefix_granularity = 2")
        .replace("block_granularity=16", "block_granularity=2")
        .replace("max_scheduled_tokens = 512", "max_scheduled_tokens = 8")
        .replace("num_device_pages = 64", "num_device_pages = 32")
        .replace("total_pages=64", "total_pages=32");
    let mut scheduler = Python::attach(|py| Scheduler::from_python(&run(py, &code)).unwrap());
    let mut queued = spec("queued", vec![100, 101]);
    queued.max_new_tokens = 1;
    scheduler
        .submit_requests(&[spec("long", (1..=52).collect()), queued])
        .unwrap();

    assert_eq!(scheduler.sizes().pending_prefill_tokens, 54);
    for prefix in (0..52).step_by(8) {
        let probe = scheduler.probe_prefill_gate();
        assert!(probe.wants_prefill, "prefix {prefix}");
        let plan = scheduler.next_execution_plan(probe.wants_prefill, true);
        let batches = plan.forward_batches();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].request_ids, ["long"]);
        assert_eq!(batches[0].extend_prefix_lens, [prefix]);
        assert_eq!(batches[0].input_lengths, [(52 - prefix).min(8)]);
        assert_eq!(
            scheduler.sizes().pending_prefill_tokens,
            i64::from((52 - prefix - 8).max(0) + 2)
        );
        assert!(plan.aborts().is_empty());
        scheduler
            .advance(&[event(
                ForwardEventKind::ExtendResult,
                "long",
                if prefix == 48 { vec![42] } else { vec![] },
            )])
            .unwrap();
        if prefix == 0 {
            assert_eq!(scheduler.active_lcm_blocks(), 28);
            assert!(scheduler
                .next_execution_plan(false, true)
                .forward_batches()
                .iter()
                .all(|batch| batch.request_ids.is_empty()));
        }
    }
    assert!(!scheduler.probe_prefill_gate().wants_prefill);
    assert!(scheduler.probe_prefill_gate().has_decode_work);
    // A peer opening a window still cannot admit the queued newcomer.
    let plan = scheduler.next_execution_plan(true, true);
    assert_eq!(plan.forward_batches()[0].request_ids, ["long"]);
    assert_eq!(plan.forward_batches()[0].num_extends, 0);
    scheduler
        .advance(&[event(ForwardEventKind::ExtendResult, "long", vec![43])])
        .unwrap();
    // With delay disabled the same request fits and follows ordinary admission.
    let plan = scheduler.next_execution_plan(true, false);
    assert_eq!(plan.forward_batches()[0].request_ids, ["queued"]);
}

#[test]
fn pending_prefill_tokens_use_the_queued_prompts_observed_cache_hit() {
    interpreter();
    let code = NEW_SCHEDULER.replace(
        "max_scheduled_tokens = 512",
        "max_scheduled_tokens = 64\ncfg.enable_short_prefill_first = True",
    );
    let mut scheduler = Python::attach(|py| Scheduler::from_python(&run(py, &code)).unwrap());
    scheduler
        .submit_requests(&[spec("seed", (0..32).collect())])
        .unwrap();
    scheduler.next_execution_plan(true, false);
    scheduler
        .advance(&[
            event(ForwardEventKind::ExtendResult, "seed", vec![42]),
            event(ForwardEventKind::Finish, "seed", Vec::new()),
        ])
        .unwrap();
    scheduler.next_execution_plan(true, false);
    scheduler
        .submit_requests(&[
            spec("cold", (1000..1064).collect()),
            spec("hot", (0..128).collect()),
        ])
        .unwrap();
    assert_eq!(scheduler.sizes().pending_prefill_tokens, 192);
    let plan = scheduler.next_execution_plan(true, false);
    assert_eq!(plan.forward_batches()[0].request_ids, ["cold"]);
    assert_eq!(scheduler.sizes().waiting, 1);
    assert_eq!(scheduler.sizes().pending_prefill_tokens, 96);
    scheduler
        .advance(&[event(ForwardEventKind::Abort, "hot", Vec::new())])
        .unwrap();
    assert_eq!(scheduler.sizes().pending_prefill_tokens, 0);
}

#[test]
fn a_request_runs_from_prefill_to_finish() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());

    scheduler
        .submit_requests(&[spec("r1", (1..=10).collect())])
        .unwrap();
    let sizes = scheduler.sizes();
    assert_eq!((sizes.waiting, sizes.running), (1, 0));
    assert_eq!(sizes.pending_prefill_tokens, 10);

    let plan = scheduler.next_execution_plan(true, false);
    let batches = plan.forward_batches();
    assert_eq!(batches.len(), 1);
    assert_eq!(scheduler.sizes().running, 1);
    assert_eq!(scheduler.sizes().pending_prefill_tokens, 0);
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
    let decode = scheduler.next_execution_plan(true, false);
    let batches = decode.forward_batches();
    assert_eq!(
        (batches[0].num_extends, batches[0].input_lengths.clone()),
        (0, vec![1])
    );
    assert_eq!(scheduler.sizes().decoding, 1);
    assert_eq!(scheduler.sizes().running, 1);

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
    scheduler.next_execution_plan(true, false);
    let sizes = scheduler.sizes();
    assert_eq!(
        (
            sizes.waiting,
            sizes.prefilling,
            sizes.decoding,
            sizes.running
        ),
        (0, 0, 0, 0)
    );
}

#[test]
fn a_recording_sees_the_schedulers_phases_inside_its_calls() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());

    profile::start();
    // Other tests share the process: this thread's spans are the marker's.
    drop(profile::span("marker"));
    scheduler
        .submit_requests(&[spec("r1", (1..=10).collect())])
        .unwrap();
    scheduler.next_execution_plan(true, false);
    scheduler
        .advance(&[event(ForwardEventKind::ExtendResult, "r1", vec![42])])
        .unwrap();
    // A decode step: its result is no prefill result.
    scheduler.next_execution_plan(true, false);
    scheduler
        .advance(&[event(ForwardEventKind::ExtendResult, "r1", vec![43])])
        .unwrap();
    let recording = profile::stop();

    let tid = recording
        .spans
        .iter()
        .find(|span| span.name == "marker")
        .expect("the marker span")
        .tid;
    let spans: Vec<_> = recording
        .spans
        .iter()
        .filter(|span| span.tid == tid)
        .collect();
    let named = |name: &str| {
        spans
            .iter()
            .find(|span| span.name == name)
            .unwrap_or_else(|| panic!("no {name} span in {spans:#?}"))
    };
    let within = |inner: &profile::Span, outer: &profile::Span| {
        outer.begin_ns <= inner.begin_ns && inner.end_ns <= outer.end_ns
    };
    let submit = named("Scheduler::submit_requests");
    assert_eq!(submit.category, profile::HOST);
    // With prefix caching on, a request's prompt page hashes are computed in
    // the round it arrives, so ComputePrefixHashes runs under submit.
    for phase in [
        "copyRequestSpecs",
        "Scheduler::SubmitRequests",
        "ComputePrefixHashes",
    ] {
        let phase = named(phase);
        assert_eq!(phase.category, profile::SCHEDULER, "{phase:?}");
        assert!(within(phase, submit), "{phase:?} outside {submit:?}");
    }
    // The control thread's call, and inside it the C++ scheduler's own spans
    // under their category.
    let plan = named("Scheduler::next_execution_plan");
    assert_eq!(plan.category, profile::HOST);
    let planned = named("Scheduler::NextExecutionPlan");
    assert!(within(planned, plan), "{planned:?} outside {plan:?}");
    for phase in [
        "Scheduler::NextExecutionPlan",
        "Scheduler::buildForwardOperations",
        "Scheduler::buildFusedPlan",
        "Scheduler::scheduleLocalPrefillWork",
        "Scheduler::schedulePrefillFirstChunk",
        "Scheduler::matchPrefixAtAdmission",
        "PrefixProber::ProbePromptPrefix",
        "CacheCoordinator::Admit",
        "planAdmission",
        "AdmissionPlanner::fitWithEviction",
        "CacheCoordinator::acquirePrefix",
        "CacheCoordinator::CollectDeadEntries",
        "ForwardBatch::ForwardBatch",
    ] {
        let phase = named(phase);
        assert_eq!(phase.category, profile::SCHEDULER, "{phase:?}");
        assert!(within(phase, planned), "{phase:?} outside {planned:?}");
    }
    let advance = named("Scheduler::advance");
    assert_eq!(advance.category, profile::HOST);
    let advanced = named("Scheduler::Advance");
    let settle = named("Scheduler::settleFinishedRequests");
    assert!(within(advanced, advance) && within(settle, advanced));
    let prefill_results: Vec<_> = spans
        .iter()
        .filter(|span| span.name == "Scheduler::handlePrefillResult")
        .collect();
    assert_eq!(prefill_results.len(), 1, "{spans:#?}");
    assert!(within(prefill_results[0], advanced));
    assert!(recording.threads.iter().any(|(thread, _)| *thread == tid));
}

#[test]
fn an_event_the_state_does_not_allow_is_an_error() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());
    scheduler
        .submit_requests(&[spec("r0", vec![1, 2, 3, 4])])
        .unwrap();
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
    scheduler
        .submit_requests(&[spec("r1", (1..=10).collect())])
        .unwrap();

    // Decode until the scheduler retracts the request: its plan stores the
    // image and runs no forward for it.
    let mut store = None;
    for token in 42..50 {
        let plan = scheduler.next_execution_plan(true, false);
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
        let plan = scheduler.next_execution_plan(true, false);
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
cfg.max_resident_requests = 4
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
    scheduler
        .submit_requests(&[spec("waiting", vec![1, 2, 3, 4])])
        .unwrap();

    // The prompt's probe runs on the storage executors: the device's cache
    // ops never count it, and the request waits for its answer.
    let plan = scheduler.next_execution_plan(true, false);
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
    let plan = scheduler.next_execution_plan(true, false);
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
    scheduler
        .submit_requests(&[spec("waiting", vec![1, 2, 3, 4])])
        .unwrap();
    let query = scheduler.next_execution_plan(true, false).storage_ops()[0].clone();

    let detach = StorageRound {
        cancellations: vec![StorageCancellationView {
            request_id: "waiting".to_owned(),
            abort: false,
        }],
        ..tick(2)
    };
    scheduler.advance_head(&detach, &[]).unwrap();
    scheduler
        .submit_requests(&[spec("next", vec![5, 6, 7, 8])])
        .unwrap();
    // The cancel goes out before the next request's probe.
    let ops = scheduler.next_execution_plan(true, false).storage_ops();
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
    scheduler
        .submit_requests(&[spec("r1", (1..=40).collect())])
        .unwrap();
    scheduler.next_execution_plan(true, false);
    scheduler
        .advance(&[event(ForwardEventKind::PdBootstrapped, "r1", Vec::new())])
        .unwrap();
    let plan = scheduler.next_execution_plan(true, false);
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
    let sizes = scheduler.sizes();
    assert_eq!(
        (sizes.remote_prefilling, sizes.prefilling, sizes.running),
        (1, 1, 0)
    );
    assert_eq!(python_transfer_view(plan, "remote_prefill"), rows);
}

#[test]
fn decode_slot_counts_distinguish_resident_ready_and_draining_work() {
    interpreter();
    let code = pd_scheduler("D")
        .replace("max_batch_size = 8", "max_batch_size = 2")
        .replace("max_resident_requests = 8", "max_resident_requests = 4");
    let mut scheduler = Python::attach(|py| Scheduler::from_python(&run(py, &code)).unwrap());
    scheduler
        .submit_requests(&[
            spec("a", vec![1, 2]),
            spec("b", vec![3, 4]),
            spec("c", vec![5, 6]),
            spec("d", vec![7, 8]),
            spec("queued", vec![9, 10]),
        ])
        .unwrap();
    for rid in ["a", "b", "c", "d", "queued"] {
        scheduler
            .advance(&[event(ForwardEventKind::PdBootstrapped, rid, vec![])])
            .unwrap();
    }
    for rid in ["a", "b", "c", "d"] {
        let transfer = scheduler
            .next_execution_plan(true, false)
            .remote_prefill_transfer()
            .unwrap();
        assert_eq!(transfer.request_ids, [rid]);
    }
    let sizes = scheduler.sizes();
    assert_eq!(
        (
            sizes.allocated_request_slots,
            sizes.active_decode_slots,
            sizes.ready_waiting
        ),
        (4, 0, 0)
    );
    // The old load counters keep their existing meanings: unadmitted work
    // waits, and remote prefills do not count as a running decode batch.
    assert_eq!(
        (sizes.running, sizes.waiting, sizes.remote_prefilling),
        (0, 1, 4)
    );
    assert!(scheduler
        .next_execution_plan(true, false)
        .remote_prefill_transfer()
        .is_none());
    for rid in ["a", "b", "c", "d"] {
        scheduler
            .advance(&[ForwardEvent {
                bootstrap_token: 42,
                ..event(ForwardEventKind::PdRemotePrefillDone, rid, vec![])
            }])
            .unwrap();
    }
    assert_eq!(scheduler.sizes().ready_waiting, 4);
    let plan = scheduler.next_execution_plan(true, false);
    assert_eq!(plan.forward_batches()[0].request_ids, ["a", "b"]);
    let sizes = scheduler.sizes();
    assert_eq!(
        (
            sizes.allocated_request_slots,
            sizes.active_decode_slots,
            sizes.ready_waiting
        ),
        (4, 2, 2)
    );
    assert_eq!(
        (sizes.running, sizes.waiting, sizes.remote_prefilling),
        (4, 1, 0),
        "ready requests retain their existing contribution to the load report"
    );
    scheduler
        .advance(&[event(ForwardEventKind::Abort, "a", vec![])])
        .unwrap();
    let sizes = scheduler.sizes();
    assert_eq!(
        (
            sizes.allocated_request_slots,
            sizes.active_decode_slots,
            sizes.ready_waiting
        ),
        (4, 2, 2),
        "an aborted in-flight forward still owns its request slot and decode permit"
    );
    scheduler
        .advance(&[event(ForwardEventKind::ExtendResult, "a", vec![43])])
        .unwrap();
    let sizes = scheduler.sizes();
    assert_eq!(
        (
            sizes.allocated_request_slots,
            sizes.active_decode_slots,
            sizes.ready_waiting
        ),
        (3, 1, 2)
    );
}

#[test]
fn a_remote_partial_prefix_has_a_live_transfer_destination() {
    interpreter();
    for prefix in [7, 23, 39] {
        let mut scheduler =
            Python::attach(|py| Scheduler::from_python(&run(py, &pd_scheduler("D"))).unwrap());
        let mut seed = spec("seed", (1..=prefix).collect());
        seed.max_new_tokens = 1;
        scheduler.submit_requests(&[seed]).unwrap();
        scheduler
            .advance(&[event(ForwardEventKind::PdBootstrapped, "seed", vec![])])
            .unwrap();
        let first = scheduler
            .next_execution_plan(true, false)
            .remote_prefill_transfer()
            .unwrap();
        let source = first.block_tables[0].blocks[(prefix / 16) as usize];
        scheduler
            .advance(&[
                ForwardEvent {
                    bootstrap_token: 1000,
                    ..event(ForwardEventKind::PdRemotePrefillDone, "seed", vec![])
                },
                event(ForwardEventKind::Finish, "seed", vec![]),
            ])
            .unwrap();
        scheduler.next_execution_plan(true, false);

        scheduler
            .submit_requests(&[spec("hit", (1..=prefix + 2).collect())])
            .unwrap();
        scheduler
            .advance(&[event(ForwardEventKind::PdBootstrapped, "hit", vec![])])
            .unwrap();
        let plan = scheduler.next_execution_plan(true, false);
        let rows = plan.remote_prefill_transfer().unwrap();
        assert_eq!(rows.extend_prefix_lens, [prefix]);
        let blocks = &rows.block_tables[0].blocks;
        let tail = (prefix / 16) as usize;
        assert!(blocks[..=tail].iter().all(|&block| block > 0), "{rows:?}");
        assert_ne!(blocks[tail], source, "the peer writes a private tail block");
        assert_eq!(python_transfer_view(plan, "remote_prefill"), rows);
    }
}

#[test]
fn a_remote_decode_carries_the_first_token_and_the_prompt_blocks() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, &pd_scheduler("P"))).unwrap());
    scheduler
        .submit_requests(&[spec("r1", (1..=20).collect())])
        .unwrap();
    scheduler.next_execution_plan(true, false);
    scheduler
        .advance(&[event(ForwardEventKind::PdBootstrapped, "r1", Vec::new())])
        .unwrap();
    let prefill = scheduler.next_execution_plan(true, false);
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
    let plan = scheduler.next_execution_plan(true, false);
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

#[test]
fn session_demotion_preserves_active_pages_and_reusable_cache() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());
    let mut request = spec("tagged", (1..=40).collect());
    request.session_id = "session-a".into();
    scheduler.submit_requests(&[request]).unwrap();
    scheduler.next_execution_plan(true, false);
    scheduler
        .advance(&[event(ForwardEventKind::ExtendResult, "tagged", vec![42])])
        .unwrap();
    let before = scheduler.active_lcm_blocks();
    scheduler.demote_session("session-a", true);
    assert_eq!(scheduler.active_lcm_blocks(), before);
    scheduler
        .advance(&[event(ForwardEventKind::Finish, "tagged", vec![])])
        .unwrap();
    scheduler.next_execution_plan(true, false);
    assert_eq!(scheduler.active_lcm_blocks(), 0);
    // Finish publishes the pages at their already-demoted session's age.
    assert_eq!(scheduler.demote_session("session-a", true), 0);
    let mut resumed = spec("resumed", (1..=40).collect());
    resumed.session_id = "session-b".into();
    scheduler.submit_requests(&[resumed]).unwrap();
    let plan = scheduler.next_execution_plan(true, false);
    assert!(
        plan.forward_batches()[0].extend_prefix_lens[0] > 0,
        "demotion must not invalidate reusable KV"
    );
}

#[test]
fn session_demotion_counts_only_newly_aged_idle_pages() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());
    let mut request = spec("tagged", (1..=40).collect());
    request.session_id = "session-a".into();
    scheduler.submit_requests(&[request]).unwrap();
    scheduler.next_execution_plan(true, false);
    scheduler
        .advance(&[event(ForwardEventKind::ExtendResult, "tagged", vec![42])])
        .unwrap();
    scheduler
        .advance(&[event(ForwardEventKind::Finish, "tagged", vec![])])
        .unwrap();
    scheduler.next_execution_plan(true, false);
    assert_eq!(scheduler.active_lcm_blocks(), 0);
    assert!(scheduler.demote_session("session-a", false) > 0);
    assert_eq!(scheduler.demote_session("session-a", false), 0);
    assert_eq!(scheduler.demote_session("session-a", true), 0);
}

#[test]
fn cache_group_ownership_survives_cut_end_and_keeps_content_matching() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());
    let mut request = spec("cut-one", (1..=40).collect());
    request.session_id = "cut-session".into();
    request.cache_group_id = "teacher-group".into();
    scheduler.submit_requests(&[request]).unwrap();
    scheduler.next_execution_plan(true, false);
    scheduler
        .advance(&[event(ForwardEventKind::ExtendResult, "cut-one", vec![42])])
        .unwrap();
    scheduler
        .advance(&[event(ForwardEventKind::Finish, "cut-one", vec![])])
        .unwrap();
    scheduler.next_execution_plan(true, false);
    assert_eq!(scheduler.demote_session("cut-session", true), 0);
    assert_eq!(scheduler.demote_session("teacher-group", true), 0);
    assert!(scheduler.demote_cache_groups(&["teacher-group".into()], true) > 0);
    let mut resumed = spec("cut-two", (1..=40).collect());
    resumed.session_id = "different-session".into();
    resumed.cache_group_id = "teacher-group".into();
    scheduler.submit_requests(&[resumed]).unwrap();
    let plan = scheduler.next_execution_plan(true, false);
    assert!(plan.forward_batches()[0].extend_prefix_lens[0] > 0);
}

#[test]
fn admission_capacity_uses_native_geometry_and_live_slots() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());
    let initial = scheduler.admission_snapshot(8);
    assert_eq!(initial.free_slots, initial.total_slots);
    assert_eq!(initial.available_blocks, initial.total_blocks);
    assert!(!initial.capacity_blocks.is_empty());
    assert!(initial
        .capacity_blocks
        .windows(2)
        .all(|pair| pair[0] <= pair[1]));
    assert!(*initial.capacity_blocks.last().unwrap() <= i64::from(initial.total_blocks));
    scheduler
        .submit_requests(&[spec("capacity", (1..=10).collect())])
        .unwrap();
    scheduler.next_execution_plan(true, false);
    let active = scheduler.admission_snapshot(8);
    assert_eq!(active.free_slots, initial.free_slots - 1);
    assert!(active.available_blocks < initial.available_blocks);
}

#[test]
fn shared_prefix_publication_without_storage_is_explicit_and_bounded() {
    interpreter();
    Python::attach(|py| {
        let object = run(py, NEW_SCHEDULER);
        let scheduler = Scheduler::from_python(&object).unwrap();
        assert!(!scheduler.shared_prefix_storage_enabled());
        let result = scheduler.shared_prefix_publication(&[1, 2, 3], 2).unwrap();
        assert_eq!(result.status, "unsupported");
        assert_eq!(result.confirmed_prefix_tokens, 0);
        assert!(scheduler.shared_prefix_publication(&[1, 2, 3], 4).is_err());
        assert!(scheduler.shared_prefix_publication(&[1, 2, 3], 0).is_err());
    });
}

#[test]
fn shared_prefix_publication_exposes_pressure_and_flush_epoch() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, STORAGE_SCHEDULER)).unwrap());
    assert!(scheduler.shared_prefix_storage_enabled());
    let first = scheduler
        .shared_prefix_publication(&[1, 2, 3, 4], 4)
        .unwrap();
    assert_eq!(first.status, "pending");
    assert_eq!(first.confirmed_prefix_tokens, 0);
    let mut pressure = tick(1);
    pressure.tick.as_mut().unwrap().queued_operations = 100;
    scheduler.advance_head(&pressure, &[]).unwrap();
    assert_eq!(
        scheduler
            .shared_prefix_publication(&[1, 2, 3, 4], 4)
            .unwrap()
            .status,
        "pressure"
    );
    assert!(scheduler.clear_cache());
    assert!(
        scheduler
            .shared_prefix_publication(&[1, 2, 3, 4], 4)
            .unwrap()
            .cache_epoch
            > first.cache_epoch
    );
    scheduler.advance_head(&tick(2), &[]).unwrap();
    assert_eq!(
        scheduler
            .shared_prefix_publication(&[1, 2, 3, 4], 4)
            .unwrap()
            .status,
        "pending"
    );
    assert_eq!(
        scheduler
            .shared_prefix_publication(&[1; 12], 12)
            .unwrap()
            .status,
        "too_large"
    );
}

#[test]
fn shared_prefix_publication_rejects_in_forward_state_before_waiting() {
    interpreter();
    for role in ["P", "Fused"] {
        for (state_grain, in_forward) in [
            (None, false),
            (None, true),
            (Some(1), false),
            (Some(1), true),
            (Some(4), true),
        ] {
            let state = state_grain.map_or_else(String::new, |grain| format!(
                "cfg.cache_groups = [*cfg.cache_groups, ts.CacheGroupConfig('state', 0, 32, ts.AttnKind.State, transfer_policy=ts.CacheTransferPolicy.LatestSnapshot, checkpoint_granularity={grain})]\ncfg.host_group_packing = [1, 1]\n"
            ));
            let enabled = if in_forward { "True" } else { "False" };
            let code = STORAGE_SCHEDULER.replace(
                "result = ts.Scheduler(cfg)",
                &format!(
                    "cfg.role = ts.SchedulerConfig.Role.{role}\ncfg.prefix_granularity = 4\ncfg.num_host_pages = 32\ncfg.storage_config = ts.StorageConfig(8, 64, 16, 64, 32, 1, 1000, 1000, ts.StoragePressure(8, 65536, 1000), ts.StoragePressure(4, 32768, 500))\ncfg.cache_groups = [ts.CacheGroupConfig('full', 2, 32, ts.AttnKind.Full, transfer_policy=ts.CacheTransferPolicy.FullSuffix)]\ncfg.state_checkpoint_in_forward = {enabled}\n{state}result = ts.Scheduler(cfg)"
                ),
            );
            let mut scheduler =
                Python::attach(|py| Scheduler::from_python(&run(py, &code)).unwrap());
            let supported = !(state_grain.is_some() && in_forward);
            // A router must detect the incompatible mode before submitting a group.
            assert_eq!(scheduler.shared_prefix_storage_enabled(), supported);
            let tokens: Vec<i32> = (1..=22).collect();
            for pressure in [false, true] {
                let mut round = tick(if pressure { 2 } else { 1 });
                round.tick.as_mut().unwrap().queued_operations = if pressure { 100 } else { 0 };
                scheduler.advance_head(&round, &[]).unwrap();
                // With page=4 and budget=8, in-forward State can save 20 and
                // suppress 22. Neither query may start an impossible wait in
                // that mode; callers need an actual-boundary receipt contract.
                for boundary in [20, 22] {
                    let result = scheduler
                        .shared_prefix_publication(&tokens, boundary)
                        .unwrap();
                    let expected = if !supported {
                        "unsupported"
                    } else if pressure {
                        "pressure"
                    } else {
                        "pending"
                    };
                    assert_eq!(
                        result.status, expected,
                        "role={role}, grain={state_grain:?}, in_forward={in_forward}, boundary={boundary}, pressure={pressure}"
                    );
                    assert_eq!(result.confirmed_prefix_tokens, 0);
                }
            }
        }
    }
}

#[test]
fn shared_prefix_publication_follows_real_prefill_host_copy_and_put() {
    interpreter();
    for state_group in [false, true] {
        let state = if state_group {
            "cfg.cache_groups = [*cfg.cache_groups, ts.CacheGroupConfig('state', 0, 32, ts.AttnKind.State, transfer_policy=ts.CacheTransferPolicy.LatestSnapshot)]\ncfg.host_group_packing = [1, 1]\n"
        } else {
            ""
        };
        let code = STORAGE_SCHEDULER.replace("result = ts.Scheduler(cfg)", &format!(
            "cfg.role = ts.SchedulerConfig.Role.P\ncfg.num_host_pages = 32\ncfg.cache_groups = [ts.CacheGroupConfig('full', 2, 32, ts.AttnKind.Full, transfer_policy=ts.CacheTransferPolicy.FullSuffix)]\ncfg.storage_config = ts.StorageConfig(8, 64, 16, 64, 32, 1, 1000, 1000, ts.StoragePressure(8, 65536, 1000), ts.StoragePressure(4, 32768, 500))\n{state}result = ts.Scheduler(cfg)"));
        let mut scheduler = Python::attach(|py| Scheduler::from_python(&run(py, &code)).unwrap());
        let tokens = vec![1, 2, 3, 4, 5]; // Includes a real partial-page prompt endpoint.
        scheduler.advance_head(&tick(1), &[]).unwrap();
        scheduler
            .submit_requests(&[spec("producer", tokens.clone())])
            .unwrap();
        let initial = scheduler.next_execution_plan(true, false);
        let mut initial_done = tick(1);
        for op in initial.storage_ops() {
            assert_eq!(op.kind, StorageOpKind::Query);
            initial_done.done.push(StorageDoneView {
                op_id: op.op_id,
                success: vec![0; op.keys.len()],
                retired: true,
                object_failures: vec![0; op.keys.len()],
            });
        }
        scheduler.advance_head(&initial_done, &[]).unwrap();
        scheduler
            .advance(&[event(ForwardEventKind::PdBootstrapped, "producer", vec![])])
            .unwrap();
        let (mut prefills, mut pd, mut stores, mut puts) = (0, 0, 0, 0);
        let mut stored_keys = Vec::new();
        for step in 2..30 {
            let plan = scheduler.next_execution_plan(true, false);
            let mut round = tick(step);
            let mut copies = Vec::new();
            for op in plan.cache_ops() {
                assert_eq!(op.kind, CacheOpKind::WriteBack);
                stores += 1;
                assert_ne!(
                    scheduler
                        .shared_prefix_publication(&tokens, 5)
                        .unwrap()
                        .status,
                    "published"
                );
                copies.push(CacheEvent {
                    kind: CacheEventKind::WriteBackDone,
                    op_id: op.op_id,
                });
            }
            for op in plan.storage_ops() {
                let succeeded = match op.kind {
                    StorageOpKind::Query => false,
                    StorageOpKind::Put => {
                        puts += 1;
                        stored_keys.extend(op.keys.clone());
                        assert_ne!(
                            scheduler
                                .shared_prefix_publication(&tokens, 5)
                                .unwrap()
                                .status,
                            "published"
                        );
                        true
                    }
                    other => panic!("unexpected operation {other:?}"),
                };
                round.done.push(StorageDoneView {
                    op_id: op.op_id,
                    success: vec![u8::from(succeeded); op.keys.len()],
                    retired: true,
                    object_failures: vec![0; op.keys.len()],
                });
            }
            for batch in plan.forward_batches() {
                for id in batch.request_ids {
                    prefills += 1;
                    scheduler
                        .advance(&[event(ForwardEventKind::ExtendResult, &id, vec![42])])
                        .unwrap();
                }
            }
            if let Some(batch) = plan.remote_decode_transfer() {
                for id in batch.request_ids {
                    pd += 1;
                    scheduler
                        .advance(&[event(ForwardEventKind::PdSucceeded, &id, vec![])])
                        .unwrap();
                }
            }
            scheduler.advance_head(&round, &copies).unwrap();
            let receipt = scheduler.shared_prefix_publication(&tokens, 5).unwrap();
            if receipt.status == "published" {
                assert_eq!(receipt.confirmed_prefix_tokens, 5);
                break;
            }
        }
        assert!(
            prefills > 0 && pd > 0 && stores > 0 && puts > 0,
            "prefill={prefills}, pd={pd}, d2h={stores}, put={puts}"
        );
        assert_eq!(
            scheduler
                .shared_prefix_publication(&tokens, 5)
                .unwrap()
                .status,
            "published",
            "state={state_group}"
        );
        assert_eq!(
            scheduler
                .shared_prefix_publication(&[1, 2, 3, 4, 9], 5)
                .unwrap()
                .status,
            "pending"
        );

        // A separate P has no producer-local tail table. The hint discovers
        // the actual unaligned checkpoint through its own Query/GET/load path.
        let mut consumer = Python::attach(|py| Scheduler::from_python(&run(py, &code)).unwrap());
        let mut request = spec("consumer", vec![1, 2, 3, 4, 5, 6, 7]);
        request.shared_prefix_tokens = vec![5];
        consumer.advance_head(&tick(1), &[]).unwrap();
        consumer.submit_requests(&[request]).unwrap();
        let mut initial = Some(consumer.next_execution_plan(true, false));
        consumer
            .advance(&[event(ForwardEventKind::PdBootstrapped, "consumer", vec![])])
            .unwrap();
        let (mut got, mut loaded, mut hit) = (false, false, 0);
        for step in 2..30 {
            let plan = initial
                .take()
                .unwrap_or_else(|| consumer.next_execution_plan(true, false));
            let mut round = tick(step);
            let mut copies = Vec::new();
            for op in plan.storage_ops() {
                assert!(matches!(op.kind, StorageOpKind::Query | StorageOpKind::Get));
                let success: Vec<u8> = op
                    .keys
                    .iter()
                    .map(|key| u8::from(stored_keys.contains(key)))
                    .collect();
                if op.kind == StorageOpKind::Get {
                    got = true;
                    assert!(success.iter().all(|&value| value == 1));
                }
                round.done.push(StorageDoneView {
                    op_id: op.op_id,
                    success,
                    retired: true,
                    object_failures: vec![0; op.keys.len()],
                });
            }
            for op in plan.cache_ops() {
                assert_eq!(op.kind, CacheOpKind::LoadBack);
                loaded = true;
                copies.push(CacheEvent {
                    kind: CacheEventKind::LoadBackDone,
                    op_id: op.op_id,
                });
            }
            for batch in plan.forward_batches() {
                if !batch.request_ids.is_empty() {
                    hit = batch.extend_prefix_lens[0];
                }
            }
            consumer.advance_head(&round, &copies).unwrap();
            if hit > 0 {
                break;
            }
        }
        assert!(
            got && loaded,
            "consumer must perform a real GET and Host load"
        );
        assert_eq!(
            hit, 5,
            "separate P must resume at the hinted partial tail, state={state_group}"
        );
        // Accepted GETs are evidence for the old shared history. Publishing
        // the longer prompt must not require re-PUT of every inherited page.
        assert_eq!(
            consumer
                .shared_prefix_publication(&tokens, 5)
                .unwrap()
                .status,
            "published"
        );
        consumer
            .advance(&[event(ForwardEventKind::ExtendResult, "consumer", vec![42])])
            .unwrap();
        for step in 30..60 {
            let plan = consumer.next_execution_plan(true, false);
            let mut round = tick(step);
            let mut copies = Vec::new();
            if let Some(batch) = plan.remote_decode_transfer() {
                for id in batch.request_ids {
                    consumer
                        .advance(&[event(ForwardEventKind::PdSucceeded, &id, vec![])])
                        .unwrap();
                }
            }
            for op in plan.cache_ops() {
                assert_eq!(op.kind, CacheOpKind::WriteBack);
                copies.push(CacheEvent {
                    kind: CacheEventKind::WriteBackDone,
                    op_id: op.op_id,
                });
            }
            for op in plan.storage_ops() {
                assert_eq!(op.kind, StorageOpKind::Put);
                round.done.push(StorageDoneView {
                    op_id: op.op_id,
                    success: vec![1; op.keys.len()],
                    retired: true,
                    object_failures: vec![0; op.keys.len()],
                });
            }
            consumer.advance_head(&round, &copies).unwrap();
            if consumer
                .shared_prefix_publication(&[1, 2, 3, 4, 5, 6, 7], 7)
                .unwrap()
                .status
                == "published"
            {
                break;
            }
        }
        assert_eq!(
            consumer
                .shared_prefix_publication(&[1, 2, 3, 4, 5, 6, 7], 7)
                .unwrap()
                .status,
            "published",
            "next producer state={state_group}"
        );
    }
}

#[test]
fn rejected_forward_conversion_does_not_apply_a_partial_batch() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());
    scheduler
        .submit_requests(&[spec("r", vec![1, 2, 3])])
        .unwrap();
    let plan = scheduler.next_execution_plan(true, false);
    assert_eq!(plan.forward_batches()[0].request_ids, ["r"]);
    let landed = event(ForwardEventKind::ExtendResult, "r", vec![7]);
    let unsupported = event(ForwardEventKind { repr: 255 }, "r", Vec::new());
    assert!(scheduler.advance(&[landed.clone(), unsupported]).is_err());
    // Conversion is completed before any event is applied. The failed call
    // does not apply the successfully converted prefix of the input.
    assert_eq!(scheduler.request_token_size("r"), 3);
    scheduler
        .advance(&[landed, event(ForwardEventKind::Finish, "r", Vec::new())])
        .unwrap();
    assert_eq!(scheduler.active_lcm_blocks(), 0);
    assert!(!scheduler.next_execution_plan(true, false).needs_device());
    // A following call cannot replay events from the failed conversion.
    scheduler.advance(&[]).unwrap();
    assert_eq!(scheduler.sizes().running, 0);
}

#[test]
fn python_event_lists_keep_their_payload_after_cache_clear() {
    interpreter();
    let code = NEW_SCHEDULER.replace(
        "result = ts.Scheduler(cfg)",
        "cfg.enable_kv_cache_events = True\ncfg.disable_l2_cache = True\ncfg.enable_l3_storage = False\nresult = ts.Scheduler(cfg)",
    );
    let owner = Python::attach(|py| run(py, &code).unbind());
    let mut scheduler = Python::attach(|py| Scheduler::from_python(owner.bind(py)).unwrap());
    scheduler
        .submit_requests(&[spec("r", (1..=33).collect())])
        .unwrap();
    assert!(scheduler.next_execution_plan(true, false).needs_device());
    scheduler
        .advance(&[
            event(ForwardEventKind::ExtendResult, "r", vec![99]),
            event(ForwardEventKind::Finish, "r", Vec::new()),
        ])
        .unwrap();
    assert!(!scheduler.next_execution_plan(true, false).needs_device());
    Python::attach(|py| {
        let scope = PyDict::new(py);
        scope.set_item("scheduler", owner.bind(py)).unwrap();
        py.run(
            c"events = scheduler.drain_kv_events()\n\
              assert [e.kind for e in events] == ['BlockStored'] * 3\n\
              assert [e.medium for e in events] == [None] * 3\n\
              hashes = [e.block_hashes[0] for e in events]\n\
              assert [e.parent_block_hash for e in events] == [None, *hashes[:-1]]\n\
              assert scheduler.clear_cache()\n\
              removed = scheduler.drain_kv_events()\n\
              assert [e.kind for e in removed] == ['BlockRemoved'] * 3\n\
              assert [e.block_hashes[0] for e in removed] == hashes[::-1]\n\
              assert [t for e in events for t in e.token_ids] == list(range(1, 34))\n\
              assert [e.block_size for e in events] == [16, 16, 16]",
            Some(&scope),
            None,
        )
        .unwrap();
    });
}

#[test]
fn frozen_rounds_do_not_forward_or_abort_waiting_requests() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());
    scheduler
        .submit_requests(&[spec("paused", vec![1, 2, 3])])
        .unwrap();
    let mut frozen = tick(1);
    frozen.tick.as_mut().unwrap().allow_forward = false;
    scheduler.advance_head(&frozen, &[]).unwrap();
    // Exceed the scheduler's no-progress threshold while forwards are forbidden.
    for _ in 0..8192 {
        let plan = scheduler.next_execution_plan(false, false);
        assert!(plan.forward_batches().is_empty());
        assert!(plan.aborts().is_empty());
        assert_eq!(scheduler.sizes().waiting, 1);
    }
    scheduler.advance_head(&tick(2), &[]).unwrap();
    let plan = scheduler.next_execution_plan(true, false);
    assert!(plan.aborts().is_empty());
    assert_eq!(plan.forward_batches()[0].request_ids, ["paused"]);
}

/// A small scheduler whose Host tier is the proxy of a shared L2 pool.
const POOL_SCHEDULER: &str = r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.Fused
cfg.prefix_granularity = 2
cfg.num_device_pages = 32
cfg.num_host_pages = 65
cfg.host_group_packing = [1]
cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteThrough
cfg.num_snapshot_pages = 1
cfg.max_scheduled_tokens = 16
cfg.max_batch_size = 4
cfg.max_resident_requests = 4
cfg.disable_l2_cache = False
cfg.shared_l2_config = ts.SharedL2Config(max_writes_in_flight=2, max_keys_per_write=8)
cfg.cache_groups = [ts.CacheGroupConfig("full", 2, 32, ts.AttnKind.Full)]
result = ts.Scheduler(cfg)
"#;

/// The pool's answer to `lookup`: its prompt's first `pages` pages, pinned
/// under `ticket`, at pool pages 11, 12, ...
fn pool_answer(lookup: &SharedL2LookupView, ticket: u64, pages: usize) -> SharedL2AnswerView {
    SharedL2AnswerView {
        request_id: lookup.request_id.clone(),
        ticket,
        num_tokens: 2 * pages as i32,
        kv_tokens: 2 * pages as i32,
        tail_tokens: 0,
        tail_hash: String::new(),
        keys: lookup.prefix_hashes[..pages]
            .iter()
            .map(|hash| StorageKeyView {
                group_id: 0,
                content_hash: hash.clone(),
                page_offset: 0,
            })
            .collect(),
        buckets: vec![0; pages],
        pool_pages: (11..11 + pages as i32).collect(),
    }
}

#[test]
fn a_pool_prompt_waits_for_its_lookup_and_resumes_behind_the_hit() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, POOL_SCHEDULER)).unwrap());
    scheduler
        .submit_requests(&[spec("waiting", (1..=7).collect())])
        .unwrap();
    let lookups = scheduler.start_shared_l2_lookups();
    let [lookup] = &lookups[..] else {
        panic!("one lookup, got {lookups:?}");
    };
    // The prompt's last token is recomputed: three full pages may hit.
    assert_eq!(lookup.request_id, "waiting");
    assert_eq!((lookup.max_tokens, lookup.prefix_hashes.len()), (6, 3));
    assert_eq!(lookup.tokens, [1, 2, 3, 4, 5, 6]);
    assert!(
        scheduler.start_shared_l2_lookups().is_empty(),
        "handed out once"
    );

    // No plan runs the prompt before its answer.
    let plan = scheduler.next_execution_plan(true, false);
    assert!(!plan.needs_device());
    assert!(plan.shared_l2_ops().is_empty());

    let proxy_blocks = scheduler
        .apply_shared_l2_answers(&[pool_answer(lookup, 7, 2)])
        .unwrap();
    assert_eq!(
        proxy_blocks
            .iter()
            .map(|block| (block.group_id, block.bucket, block.pool_page))
            .collect::<Vec<_>>(),
        [(0, 0, 11), (0, 0, 12)]
    );

    // The prompt resumes behind the hit, which a load-back reads from the
    // proxy blocks.
    let plan = scheduler.next_execution_plan(true, false);
    let batch = &plan.forward_batches()[0];
    assert_eq!(batch.request_ids, ["waiting"]);
    assert_eq!(batch.extend_prefix_lens, [4]);
    let ops = plan.cache_ops();
    assert_eq!(
        ops.iter().map(|op| op.kind).collect::<Vec<_>>(),
        [CacheOpKind::LoadBack]
    );
    assert!(plan.shared_l2_ops().is_empty());

    // Once the load landed nothing holds the proxy blocks: the next plan
    // releases the ticket.
    scheduler
        .advance_head(
            &StorageRound::default(),
            &[CacheEvent {
                kind: CacheEventKind::LoadBackDone,
                op_id: ops[0].op_id,
            }],
        )
        .unwrap();
    let ops = scheduler.next_execution_plan(true, false).shared_l2_ops();
    let [release, write] = &ops[..] else {
        panic!("a release and a write, got {ops:?}");
    };
    assert_eq!((release.kind, release.ticket), (SharedL2OpKind::Release, 7));
    // What the prompt computed goes to the pool: its third page and its
    // tail, not the two pages it loaded from there.
    assert_eq!(write.kind, SharedL2OpKind::Write);
    assert_eq!(write.keys.len(), 2);
    assert_eq!(write.keys[0].content_hash, lookup.prefix_hashes[2]);
}

#[test]
fn a_pool_write_ends_with_its_completion() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, POOL_SCHEDULER)).unwrap());
    scheduler
        .submit_requests(&[spec("writer", (1..=7).collect())])
        .unwrap();
    let lookup = scheduler.start_shared_l2_lookups().remove(0);
    assert!(scheduler
        .apply_shared_l2_answers(&[pool_answer(&lookup, 0, 0)])
        .unwrap()
        .is_empty());
    scheduler.next_execution_plan(true, false);
    scheduler
        .advance(&[event(ForwardEventKind::ExtendResult, "writer", vec![9])])
        .unwrap();

    // What the prompt published goes to the pool: its three full pages and
    // its tail, each from the Device block that holds it.
    let ops = scheduler.next_execution_plan(true, false).shared_l2_ops();
    let [write] = &ops[..] else {
        panic!("one write, got {ops:?}");
    };
    assert_eq!(write.kind, SharedL2OpKind::Write);
    assert_eq!(
        write.keys[..3]
            .iter()
            .map(|key| key.content_hash.as_str())
            .collect::<Vec<_>>(),
        lookup.prefix_hashes
    );
    assert_eq!(write.buckets, [0, 0, 0, 0]);
    assert_eq!(write.device_blocks.len(), 4);
    assert!(write.checkpoints.is_empty(), "no State checkpoint");
    assert!(!scheduler.can_clear_cache());

    // Its completion joins the head batch as an acknowledgment and lets its
    // blocks go; a completion of a write that is not in flight is an error.
    let done = CacheEvent {
        kind: CacheEventKind::SharedL2WriteDone,
        op_id: write.op_id,
    };
    scheduler
        .advance_head(&StorageRound::default(), std::slice::from_ref(&done))
        .unwrap();
    let error = scheduler
        .advance_head(&StorageRound::default(), &[done])
        .unwrap_err();
    assert!(error.what().contains("unknown shared L2 write"), "{error}");
}

#[test]
fn an_answer_the_scheduler_cannot_take_is_an_error() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, POOL_SCHEDULER)).unwrap());
    scheduler
        .submit_requests(&[
            spec("missed", (1..=7).collect()),
            spec("hit", (11..=17).collect()),
        ])
        .unwrap();
    let lookups = scheduler.start_shared_l2_lookups();
    let [missed, hit] = &lookups[..] else {
        panic!("two lookups, got {lookups:?}");
    };

    // An answer that does not fit its prompt is an error, and so is one the
    // proxy cannot mirror: nothing of either is kept.
    let mut misfit = pool_answer(hit, 3, 1);
    misfit.num_tokens = 3;
    let error = scheduler.apply_shared_l2_answers(&[misfit]).unwrap_err();
    assert!(error.what().contains("outside the prompt"), "{error}");
    let mut unhashed = pool_answer(hit, 3, 1);
    unhashed.keys[0].content_hash.clear();
    let error = scheduler.apply_shared_l2_answers(&[unhashed]).unwrap_err();
    assert!(error.what().contains("names keys"), "{error}");
    // A tail is its tokens and its hash: a hash without tokens is refused.
    let mut tokenless_tail = pool_answer(hit, 3, 1);
    tokenless_tail.tail_hash = hit.prefix_hashes[0].clone();
    let error = scheduler
        .apply_shared_l2_answers(&[tokenless_tail])
        .unwrap_err();
    assert!(
        error.what().contains("tail lies strictly inside"),
        "{error}"
    );
    // No plan runs a prompt whose lookup has no answer, and none has a
    // ticket of a refused answer to release.
    let plan = scheduler.next_execution_plan(true, false);
    assert!(!plan.needs_device() && plan.shared_l2_ops().is_empty());

    // A miss and a hit settle their lookups in one call, which returns the
    // hit's proxy blocks alone; a second answer for a request names no
    // lookup that waits.
    let proxy_blocks = scheduler
        .apply_shared_l2_answers(&[pool_answer(missed, 0, 0), pool_answer(hit, 5, 2)])
        .unwrap();
    assert_eq!(
        proxy_blocks
            .iter()
            .map(|block| block.pool_page)
            .collect::<Vec<_>>(),
        [11, 12]
    );
    let error = scheduler
        .apply_shared_l2_answers(&[pool_answer(hit, 6, 2)])
        .unwrap_err();
    assert!(error.what().contains("no lookup that waits"), "{error}");
    // Neither does an answer for a request whose lookup was not handed out,
    // or for one the scheduler does not know.
    scheduler
        .submit_requests(&[spec("unasked", (21..=27).collect())])
        .unwrap();
    for request_id in ["unasked", "unknown"] {
        let answer = SharedL2AnswerView {
            request_id: request_id.to_owned(),
            ..pool_answer(missed, 0, 0)
        };
        let error = scheduler.apply_shared_l2_answers(&[answer]).unwrap_err();
        assert!(error.what().contains("no lookup that waits"), "{error}");
    }
    let plan = scheduler.next_execution_plan(true, false);
    let batch = &plan.forward_batches()[0];
    assert_eq!(batch.request_ids, ["missed", "hit"]);
    assert_eq!(batch.extend_prefix_lens, [0, 4]);
}

#[test]
fn a_plan_of_pool_commands_alone_needs_no_device() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, POOL_SCHEDULER)).unwrap());
    scheduler
        .submit_requests(&[spec("gone", (1..=7).collect())])
        .unwrap();
    let lookup = scheduler.start_shared_l2_lookups().remove(0);
    scheduler
        .apply_shared_l2_answers(&[pool_answer(&lookup, 7, 2)])
        .unwrap();
    // The request leaves before a plan admitted it: nothing holds the proxy
    // blocks of its hit, and the next plan releases the ticket.
    scheduler
        .advance(&[event(ForwardEventKind::Abort, "gone", vec![])])
        .unwrap();
    let plan = scheduler.next_execution_plan(true, false);
    assert_eq!(
        plan.shared_l2_ops()
            .iter()
            .map(|op| (op.kind, op.ticket))
            .collect::<Vec<_>>(),
        [(SharedL2OpKind::Release, 7)]
    );
    // The pool's client runs them: the data plane gets nothing.
    assert!(!plan.needs_device());
    assert!(plan.cache_ops().is_empty());
}

#[test]
fn without_a_pool_the_pool_calls_ask_nothing_and_fail_as_errors() {
    interpreter();
    let mut scheduler =
        Python::attach(|py| Scheduler::from_python(&run(py, NEW_SCHEDULER)).unwrap());
    scheduler
        .submit_requests(&[spec("plain", (1..=40).collect())])
        .unwrap();
    assert!(scheduler.start_shared_l2_lookups().is_empty());
    let answer = SharedL2AnswerView {
        request_id: "plain".to_owned(),
        ticket: 0,
        num_tokens: 0,
        kv_tokens: 0,
        tail_tokens: 0,
        tail_hash: String::new(),
        keys: Vec::new(),
        buckets: Vec::new(),
        pool_pages: Vec::new(),
    };
    let error = scheduler.apply_shared_l2_answers(&[answer]).unwrap_err();
    assert!(
        error.what().contains("without a configured pool"),
        "{error}"
    );
    let plan = scheduler.next_execution_plan(true, false);
    assert_eq!(plan.forward_batches()[0].request_ids, ["plain"]);
    assert!(plan.shared_l2_ops().is_empty());
}

/// A pool of 8 LCM blocks as its owner builds it: pages of two tokens, a
/// full-attention group with two blocks to an LCM block and a State group
/// with one.
const NEW_POOL: &str = r#"
import tokenspeed_scheduler as ts
full = ts.CacheGroupConfig()
full.group_id, full.block_granularity, full.attn_kind = "full", 2, ts.AttnKind.Full
full.cache_blocks_per_lcm_block = 2
state = ts.CacheGroupConfig()
state.group_id, state.block_granularity, state.attn_kind = "state", 0, ts.AttnKind.State
state.cache_blocks_per_lcm_block = 1
cfg = ts.SchedulerConfig()
cfg.prefix_granularity = 2
cfg.cache_groups = [full, state]
cfg.host_group_packing = [2, 1]
cfg.checkpoint_retention_requests = 0
result = ts.SharedL2Pool(cfg, 8, 0)
"#;

/// The chain hashes of the prompt 1..=5 as a rank's scheduler computes
/// them: its two full pages and its tail of one token. The pool hashes a
/// looked-up prompt's own tail to find a published one.
const POOL_PROMPT_HASHES: &str = r#"
import hashlib, struct
def page_hash(tokens, prior):
    prior = bytes.fromhex(prior)
    return hashlib.sha256(
        struct.pack("<I", len(prior)) + prior + struct.pack(f"<I{len(tokens)}i", len(tokens), *tokens)
    ).hexdigest()
first = page_hash([1, 2], "")
second = page_hash([3, 4], first)
result = [first, second, page_hash([5], second)]
"#;

fn pool_prompt_hashes() -> [String; 3] {
    Python::attach(|py| run(py, POOL_PROMPT_HASHES).extract().unwrap())
}

fn pool_key(group_id: u32, content_hash: &str) -> StorageKeyView {
    StorageKeyView {
        group_id,
        content_hash: content_hash.to_owned(),
        page_offset: 0,
    }
}

/// The blocks of the prompt 1..=5: its two full pages, its tail's block,
/// and the State checkpoint at its end.
fn pool_prompt_keys() -> Vec<StorageKeyView> {
    let [first, second, tail] = pool_prompt_hashes();
    vec![
        pool_key(0, &first),
        pool_key(0, &second),
        pool_key(0, &tail),
        pool_key(1, &tail),
    ]
}

/// The checkpoint at the end of the prompt 1..=5.
fn pool_prompt_checkpoint() -> SharedL2CheckpointView {
    let [first, second, tail] = pool_prompt_hashes();
    SharedL2CheckpointView {
        tokens: (1..=5).collect(),
        prefix_hashes: vec![first, second],
        tail_tokens: 1,
        tail_hash: tail,
        access_epoch: 7,
    }
}

/// The lookup of the prompt 1..=7, which starts with the prompt 1..=5. A
/// page past the checkpoint is in no pool: its hash is never matched.
fn pool_lookup() -> SharedL2LookupView {
    let [first, second, _] = pool_prompt_hashes();
    SharedL2LookupView {
        request_id: "longer".to_owned(),
        prefix_hashes: vec![first, second, "1-2-3-4-5-6".to_owned()],
        tokens: (1..=7).collect(),
        max_tokens: 7,
    }
}

#[test]
fn only_a_pool_object_is_accepted() {
    interpreter();
    Python::attach(|py| {
        let not_a_pool = run(py, "result = object()");
        let error = SharedL2Pool::from_python(&not_a_pool).err().unwrap();
        assert!(error.is_instance_of::<pyo3::exceptions::PyTypeError>(py));
    });
}

#[test]
fn a_pool_python_built_is_written_and_read_from_rust() {
    interpreter();
    let mut pool = Python::attach(|py| SharedL2Pool::from_python(&run(py, NEW_POOL)).unwrap());
    let empty = pool.stats();
    assert_eq!(
        empty,
        SharedL2PoolStatsView {
            num_lcm_blocks: 8,
            empty_lcm_blocks: 8,
            cached_blocks: 0,
            pinned_blocks: 0,
            reserved_blocks: 0,
            open_lookups: 0,
            evicted_blocks: 0,
            checkpoints: 0,
        }
    );
    // An empty pool answers a miss, for the request that asked.
    let miss = pool.lookup(2, &pool_lookup()).unwrap();
    assert_eq!((miss.request_id.as_str(), miss.ticket), ("longer", 0));
    assert!(miss.keys.is_empty() && miss.pool_pages.is_empty());

    // Client 1 reserves a page for every block and commits them with the
    // checkpoint.
    let keys = pool_prompt_keys();
    let reservation = pool.reserve(1, &keys, &[0; 4]).unwrap();
    assert!(reservation.ticket != 0);
    assert!(
        reservation.pages.iter().all(|page| *page >= 1),
        "{reservation:?}"
    );
    assert_eq!(pool.stats().reserved_blocks, 4);
    // Another client gets no page for a block that is being written.
    let second = pool.reserve(2, &keys, &[0; 4]).unwrap();
    assert_eq!((second.ticket, second.pages), (0, vec![-3; 4]));
    pool.commit(1, reservation.ticket, &[pool_prompt_checkpoint()])
        .unwrap();
    let stats = pool.stats();
    assert_eq!(
        (
            stats.reserved_blocks,
            stats.cached_blocks,
            stats.checkpoints
        ),
        (0, 4, 1)
    );

    // Client 2's longer prompt hits the checkpoint: its blocks at the pages
    // the write got, pinned under the answer's ticket.
    let hit = pool.lookup(2, &pool_lookup()).unwrap();
    assert!(hit.ticket != 0);
    assert_eq!((hit.num_tokens, hit.kv_tokens), (5, 5));
    assert_eq!(
        (hit.tail_tokens, &hit.tail_hash),
        (1, &pool_prompt_hashes()[2])
    );
    assert_eq!(hit.keys, keys);
    assert_eq!(hit.buckets, [0; 4]);
    assert_eq!(hit.pool_pages, reservation.pages);
    let stats = pool.stats();
    assert_eq!((stats.pinned_blocks, stats.open_lookups), (4, 1));
    assert!(!pool.clear_cache(), "a pinned pool is not emptied");

    // A ticket is its client's.
    let error = pool.release(1, hit.ticket).unwrap_err();
    assert!(
        error.what().contains("no lookup of its client holds"),
        "{error}"
    );
    pool.release(2, hit.ticket).unwrap();
    assert_eq!(pool.stats().pinned_blocks, 0);
    pool.touch(&keys[..2]).unwrap();
    assert!(pool.clear_cache());
    assert_eq!(pool.stats(), empty);
}

#[test]
fn a_pool_refuses_what_is_not_its_own_as_an_error() {
    interpreter();
    let mut pool = Python::attach(|py| SharedL2Pool::from_python(&run(py, NEW_POOL)).unwrap());
    // A group the pool does not have, a bucket outside the group's shards,
    // tokens short of the lookup's bound.
    let error = pool.reserve(1, &[pool_key(2, "1-2")], &[0]).unwrap_err();
    assert!(error.what().contains("a shared L2 key names"), "{error}");
    let error = pool.touch(&[pool_key(0, "")]).unwrap_err();
    assert!(error.what().contains("a shared L2 key names"), "{error}");
    let error = pool.reserve(1, &[pool_key(0, "1-2")], &[1]).unwrap_err();
    assert!(
        error.what().contains("outside its group's shard"),
        "{error}"
    );
    let short = SharedL2LookupView {
        max_tokens: 8,
        ..pool_lookup()
    };
    let error = pool.lookup(1, &short).unwrap_err();
    assert!(error.what().contains("must reach its bound"), "{error}");

    // A commit names a reservation of its client, and a checkpoint covers
    // its pages and tail exactly: one that ends on a page has no tail.
    let reservation = pool.reserve(1, &pool_prompt_keys(), &[0; 4]).unwrap();
    let error = pool
        .commit(2, reservation.ticket, &[pool_prompt_checkpoint()])
        .unwrap_err();
    assert!(error.what().contains("reserves nothing"), "{error}");
    let tailless = SharedL2CheckpointView {
        tail_tokens: 0,
        tail_hash: String::new(),
        ..pool_prompt_checkpoint()
    };
    let error = pool.commit(1, reservation.ticket, &[tailless]).unwrap_err();
    assert!(error.what().contains("cover its pages and tail"), "{error}");
    assert_eq!(pool.stats().reserved_blocks, 4);

    // A client that leaves loses what it reserved.
    pool.disconnect(1);
    assert_eq!(pool.stats().reserved_blocks, 0);
    let error = pool
        .commit(1, reservation.ticket, &[pool_prompt_checkpoint()])
        .unwrap_err();
    assert!(error.what().contains("reserves nothing"), "{error}");
}
