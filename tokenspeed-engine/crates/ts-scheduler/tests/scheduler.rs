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
    CacheEvent, CacheEventKind, CacheOpKind, ForwardEvent, ForwardEventKind, RequestSpec, Scheduler,
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
        .advance_cache(&[CacheEvent {
            kind: CacheEventKind::SnapshotDone,
            op_id: store,
            landed_pages: 0,
        }])
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
                .advance_cache(&[CacheEvent {
                    kind: CacheEventKind::RestoreDone,
                    op_id: op.op_id,
                    landed_pages: 0,
                }])
                .unwrap();
            restored = true;
            break;
        }
    }
    assert!(restored, "the retracted request was never restored");
    let sizes = scheduler.sizes();
    assert_eq!((sizes.decoding, sizes.retracted), (1, 0));
}
