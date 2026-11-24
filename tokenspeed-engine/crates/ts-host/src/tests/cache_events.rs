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

use std::{
    ffi::CString,
    sync::{mpsc, Mutex},
    time::{Duration, Instant},
};

use tracing_appender::non_blocking::{NonBlockingBuilder, WorkerGuard};
use ts_control::event_loop::DataPlane;
use ts_scheduler::RequestSpec;

use super::*;
use crate::data_plane::PythonDataPlane;

#[derive(Clone)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn writer() -> (NonBlocking, WorkerGuard, Capture) {
    let capture = Capture(Arc::new(Mutex::new(Vec::new())));
    let (writer, guard) = NonBlockingBuilder::default().finish(capture.clone());
    (writer, guard, capture)
}

fn records(capture: &Capture) -> Vec<Value> {
    String::from_utf8(capture.0.lock().unwrap().clone())
        .unwrap()
        .lines()
        .map(|line| {
            assert!(line.len() < MAX_RECORD_BYTES);
            serde_json::from_str(line.strip_prefix("[cache-event] ").unwrap()).unwrap()
        })
        .collect()
}

fn interpreter() {
    // Native type registration releases the GIL internally. Share the host
    // suite's one-time initialization so parallel tests cannot register twice.
    crate::data_plane::tests::interpreter();
}

fn scheduler() -> Scheduler {
    interpreter();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        globals
            .set_item("ts", ts_scheduler::python_module(py).unwrap())
            .unwrap();
        py.run(c"
cfg = ts.SchedulerConfig()
cfg.prefix_granularity = 16
cfg.max_scheduled_tokens = 64
cfg.max_batch_size = 4
cfg.max_resident_requests = 4
cfg.num_device_pages = 64
cfg.num_snapshot_pages = 1
cfg.cache_groups = [ts.CacheGroupConfig(group_id='full_attention', block_granularity=16, total_pages=64, attn_kind=ts.AttnKind.Full)]
scheduler = ts.Scheduler(cfg)
", Some(&globals), None).unwrap();
        Scheduler::from_python(&globals.get_item("scheduler").unwrap().unwrap()).unwrap()
    })
}

fn settings() -> Settings {
    Settings::parse(Some("1"), Some("test-instance"), true)
        .unwrap()
        .unwrap()
}

fn install(scheduler: &mut Scheduler, writer: NonBlocking) -> CacheEvents {
    CacheEvents::install(settings(), "null", 2, 5, writer, scheduler).unwrap()
}

fn teardown(scheduler: Scheduler, context: CacheEvents) {
    drop(context);
    drop(scheduler);
    // Match the host's PythonLogGuard: reacquire the GIL to release deferred
    // native owners; do not rely on an explicit collection or finalization.
    Python::attach(|_| {});
}

#[test]
fn opt_in_requires_bounded_identity_and_explicit_async_logging() {
    assert!(Settings::parse(None, None, false).unwrap().is_none());
    assert!(Settings::parse(Some("0"), None, true).unwrap().is_none());
    assert!(Settings::parse(Some("true"), Some("test"), true).is_err());
    assert!(Settings::parse(Some("1"), Some("test"), false).is_err());
    for instance in [
        None,
        Some(""),
        Some("a/b"),
        Some("x\ny"),
        Some("é"),
        Some(&"x".repeat(129)),
    ] {
        assert!(Settings::parse(Some("1"), instance, true).is_err());
    }
    assert!(Settings::parse(Some("1"), Some(&"a".repeat(128)), true)
        .unwrap()
        .is_some());
}

#[test]
fn native_module_reports_linked_capability_without_runtime_imports() {
    crate::tests::run_in_child(
        "cache_events::tests::native_module_reports_linked_capability_without_runtime_imports",
        || {
            interpreter();
            Python::attach(|py| {
                let host = pyo3::wrap_pymodule!(crate::_host)(py);
                let actual: String = host
                    .call_method0(py, "cache_event_protocol")
                    .unwrap()
                    .extract(py)
                    .unwrap();
                assert_eq!(actual, protocol());
                assert_eq!(ts_scheduler::cache_event_protocol_version(), 1);
                let modules = py.import("sys").unwrap().getattr("modules").unwrap();
                for name in ["torch", "tokenspeed.runtime.engine.rust_host"] {
                    assert!(!modules.contains(name).unwrap());
                }
            });
        },
    );
}

#[test]
fn scheduler_assignment_and_raii_release_share_bounded_host_identity() {
    let mut scheduler = scheduler();
    let (writer, guard, capture) = writer();
    let context = install(&mut scheduler, writer);
    scheduler
        .submit_requests(&[RequestSpec {
            request_id: "a\n\"quoted\"\\request".to_owned(),
            admission_id: 42,
            session_id: String::new(),
            cache_group_id: String::new(),
            shared_prefix_tokens: Vec::new(),
            peer_holder: String::new(),
            tokens: vec![1, 2],
            max_new_tokens: 4,
            max_cached_prefix_tokens: i32::MAX,
        }])
        .unwrap();
    drop(scheduler.next_execution_plan(true, false));
    teardown(scheduler, context);
    drop(guard);
    let records = records(&capture);
    assert_eq!(
        records
            .iter()
            .map(|v| v["event"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "trace_started",
            "slot_assigned",
            "slot_released",
            "trace_finished"
        ]
    );
    for (sequence, value) in records.iter().enumerate() {
        assert_eq!(value["record_sequence"], sequence);
        assert_eq!(value["instance_id"], "test-instance");
        assert_eq!(value["role"], "fused");
        assert_eq!(value["dp_rank"], 2);
        assert_eq!(value["physical_rank"], 5);
        assert_eq!(value["log_dropped_lines"], 0);
    }
    assert_eq!(records[1]["request_id"], "a\n\"quoted\"\\request");
    assert_eq!(records[1]["slot"], records[2]["slot"]);
    assert_eq!(records[1]["slot_generation"], records[2]["slot_generation"]);
    assert_eq!(records[2]["admission_id"], 42);
    assert_eq!(records[3]["evidence_valid"], true);
}

const COMPLETION_PLANE: &str = r#"
from contextlib import nullcontext

class CopyBoundary:
    # The real Rust host is driven against a CPU replacement for device ACKs.
    def __init__(self):
        self.ready = False
        self.pending = True
        self.polled = False
        self.invalid = False
        self.fail_observation_poll = False
        self.zero = False
        self.loss = 0
    def control_thread_guard(self): return nullcontext()
    def enter_forward_thread(self): pass
    def exit_forward_thread(self): pass
    def poll_cache_events(self):
        self.polled = True
        if not self.ready or not self.pending: return []
        self.pending = False
        return [('RestoreDoneEvent', 7)]
    def poll_cache_observations(self):
        assert self.polled
        self.polled = False
        if self.fail_observation_poll: raise RuntimeError('unavailable observation')
        if not self.ready or self.pending: return ([], 0)
        self.ready = False
        return ([dict(
            kind='snapshot_restore', op_id=7, op_generation=3,
            request_id='r' * (257 if self.invalid else 1), admission_id=42,
            request_pool_index=2, slot_generation=9, previous_slot=1,
            previous_slot_generation=4, slot_previous_admission_id=41,
            restore_sequence=6, committed_tokens=100, computed_tokens=99, prompt_tokens=60,
            l2_rows=0 if self.zero else 2, l2_bytes=0 if self.zero else 128,
            pool_rows=0 if self.zero else 3, pool_bytes=0 if self.zero else 192,
            slot_state_image_bytes=0 if self.zero else 16)], self.loss)
plane = CopyBoundary()
"#;

fn plane() -> Py<PyAny> {
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(
            &CString::new(COMPLETION_PLANE).unwrap(),
            Some(&globals),
            None,
        )
        .unwrap();
        globals.get_item("plane").unwrap().unwrap().unbind()
    })
}

#[test]
fn actual_host_preserves_ack_and_labels_completed_physical_copy() {
    for zero in [false, true] {
        let mut scheduler = scheduler();
        let (writer, guard, capture) = writer();
        let context = install(&mut scheduler, writer);
        let plane = plane();
        let mut host = PythonDataPlane::start(
            Python::attach(|py| plane.clone_ref(py)),
            None,
            Some(context.clone()),
        )
        .unwrap();
        assert!(host.poll_cache_events().unwrap().is_empty());
        Python::attach(|py| {
            plane.setattr(py, "ready", true).unwrap();
            plane.setattr(py, "zero", zero).unwrap();
        });
        let ack = host.poll_cache_events().unwrap();
        assert_eq!(ack.len(), 1);
        assert_eq!(ack[0].kind, ts_scheduler::CacheEventKind::RestoreDone);
        assert_eq!(ack[0].op_id, 7);
        assert!(host.poll_cache_events().unwrap().is_empty());
        drop(host);
        teardown(scheduler, context);
        drop(guard);
        let records = records(&capture);
        let completed: Vec<_> = records
            .iter()
            .filter(|v| v["event"] == "restore_completed")
            .collect();
        assert_eq!(completed.len(), 1);
        let event = completed[0];
        assert_eq!(event["scope"], "local_physical");
        assert_eq!(event["kind"], "snapshot_restore");
        assert_eq!(event["op_generation"], 3);
        assert_eq!(event["op_id"], 7);
        assert_eq!(event["admission_id"], 42);
        assert_eq!(event["slot"], 2);
        assert_eq!(event["physical_rank"], 5);
        assert_eq!(event["l2_rows"], if zero { 0 } else { 2 });
        assert_eq!(event["pool_bytes"], if zero { 0 } else { 192 });
        assert_eq!(event["slot_state_image_bytes"], if zero { 0 } else { 16 });
        assert_eq!(records.last().unwrap()["evidence_valid"], true);
    }
}

#[test]
fn malformed_or_lost_observation_invalidates_evidence_without_changing_ack() {
    for mode in ["invalid", "fail_observation_poll", "loss"] {
        let mut scheduler = scheduler();
        let (writer, guard, capture) = writer();
        let context = install(&mut scheduler, writer);
        let plane = plane();
        Python::attach(|py| {
            plane.setattr(py, "ready", true).unwrap();
            plane.setattr(py, mode, 1).unwrap();
        });
        let mut host = PythonDataPlane::start(plane, None, Some(context.clone())).unwrap();
        let ack = host.poll_cache_events().unwrap();
        assert_eq!(ack.len(), 1);
        assert_eq!(ack[0].op_id, 7);
        drop(host);
        teardown(scheduler, context);
        drop(guard);
        let records = records(&capture);
        assert!(records.iter().any(|v| v["event"] == "observation_loss"));
        assert_eq!(records.last().unwrap()["evidence_valid"], false);
        if mode != "loss" {
            assert!(!records.iter().any(|v| v["event"] == "restore_completed"));
        }
    }
}

#[test]
fn prefix_completion_has_same_context_without_restore_frontier() {
    interpreter();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(c"record = dict(kind='prefix_load', op_id=0, request_id='prefix', admission_id=7, request_pool_index=2, slot_generation=3, op_generation=1, l2_rows=4, l2_bytes=256, pool_rows=0, pool_bytes=0, slot_state_image_bytes=0)", Some(&globals), None).unwrap();
        let record = completion(&globals.get_item("record").unwrap().unwrap()).unwrap();
        assert_eq!(record["event"], "prefix_load_completed");
        assert_eq!(record["scope"], "local_physical");
        assert_eq!(record["slot"], 2);
        assert_eq!(record["l2_bytes"], 256);
        assert!(record.get("committed_tokens").is_none());
    });
}

#[test]
fn oversize_record_is_never_written_as_partial_json() {
    let mut scheduler = scheduler();
    let (writer, guard, capture) = writer();
    let context = install(&mut scheduler, writer);
    context
        .0
        .emit(json!({"event": "oversize", "value": "a".repeat(MAX_RECORD_BYTES)}));
    teardown(scheduler, context);
    drop(guard);
    let records = records(&capture);
    assert_eq!(records.len(), 3);
    assert_eq!(records[0]["record_sequence"], 0);
    assert_eq!(records[1]["record_sequence"], 2);
    assert_eq!(records[1]["event"], "observation_loss");
    assert_eq!(records[1]["reason"], "record_too_large");
    assert_eq!(records[2]["lost_observations"], 1);
    assert_eq!(records[2]["evidence_valid"], false);
}

#[test]
fn blocked_output_drops_are_visible_and_never_block_producer() {
    struct Stalled {
        capture: Capture,
        entered: mpsc::Sender<()>,
        resume: mpsc::Receiver<()>,
        first: bool,
    }
    impl Write for Stalled {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.first {
                self.first = false;
                self.entered.send(()).unwrap();
                self.resume.recv_timeout(Duration::from_secs(5)).unwrap();
            }
            self.capture.write(bytes)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let capture = Capture(Arc::new(Mutex::new(Vec::new())));
    let (entered, wait) = mpsc::channel();
    let (release, resume) = mpsc::channel();
    let (writer, guard) = NonBlockingBuilder::default()
        .buffered_lines_limit(4)
        .lossy(true)
        .finish(Stalled {
            capture: capture.clone(),
            entered,
            resume,
            first: true,
        });
    let mut scheduler = scheduler();
    let context = install(&mut scheduler, writer.clone());
    wait.recv_timeout(Duration::from_secs(5)).unwrap();
    for _ in 0..32 {
        context.0.emit(json!({"event": "bounded_probe"}));
    }
    assert!(writer.error_counter().dropped_lines() > 0);
    release.send(()).unwrap();
    let wait_for_lines = |count| {
        let deadline = Instant::now() + Duration::from_secs(5);
        while records(&capture).len() < count {
            assert!(Instant::now() < deadline, "logging worker did not drain");
            std::thread::yield_now();
        }
    };
    wait_for_lines(5);
    context.0.report_losses();
    wait_for_lines(6);
    teardown(scheduler, context);
    drop(guard);
    let records = records(&capture);
    let loss = records
        .iter()
        .find(|v| v["event"] == "observation_loss")
        .unwrap();
    assert_eq!(loss["reason"], "log_queue_overflow");
    assert!(loss["dropped_count"].as_u64().unwrap() > 0);
    assert_eq!(records.last().unwrap()["event"], "trace_finished");
    assert_eq!(records.last().unwrap()["evidence_valid"], false);
}

#[test]
fn rejected_native_identity_is_explicit_loss_through_scheduler_teardown() {
    let mut scheduler = scheduler();
    let (writer, guard, capture) = writer();
    let context = install(&mut scheduler, writer);
    scheduler
        .submit_requests(&[RequestSpec {
            request_id: "é".repeat(129),
            admission_id: 43,
            session_id: String::new(),
            cache_group_id: String::new(),
            shared_prefix_tokens: Vec::new(),
            peer_holder: String::new(),
            tokens: vec![1, 2],
            max_new_tokens: 4,
            max_cached_prefix_tokens: i32::MAX,
        }])
        .unwrap();
    drop(scheduler.next_execution_plan(true, false));
    teardown(scheduler, context);
    drop(guard);
    let records = records(&capture);
    let loss = records
        .iter()
        .find(|v| v["event"] == "observation_loss")
        .unwrap();
    assert_eq!(loss["reason"], "native_observation_rejected");
    assert_eq!(loss["dropped_count"], 2);
    assert_eq!(records.last().unwrap()["evidence_valid"], false);
    assert!(!records
        .iter()
        .any(|v| v["event"] == "slot_assigned" || v["event"] == "slot_released"));
}

#[path = "cache_events_shutdown.rs"]
mod shutdown;

#[test]
fn concurrent_producers_keep_sequence_in_queue_arrival_order() {
    let mut scheduler = scheduler();
    let (writer, guard, capture) = writer();
    let context = install(&mut scheduler, writer);
    let start = Arc::new(std::sync::Barrier::new(8));
    std::thread::scope(|scope| {
        for producer in 0..8 {
            let context = &context;
            let start = &start;
            scope.spawn(move || {
                start.wait();
                for index in 0..128 {
                    context.0.emit(
                        json!({"event": "concurrent_probe", "producer": producer, "index": index}),
                    );
                }
            });
        }
    });
    teardown(scheduler, context);
    drop(guard);
    let records = records(&capture);
    assert_eq!(records.len(), 1026);
    for (sequence, record) in records.iter().enumerate() {
        assert_eq!(
            record["record_sequence"], sequence,
            "sequence diverged at queue position {sequence}"
        );
    }
    assert_eq!(records.last().unwrap()["evidence_valid"], true);
}

#[test]
fn a_contended_publication_handoff_is_bounded_and_reports_overflow() {
    let mut scheduler = scheduler();
    let (writer, guard, capture) = writer();
    let context = install(&mut scheduler, writer);
    // Pause a real publisher while other observations arrive. This must not
    // make an emitter wait, including a native destructor on another thread.
    let publisher = context.0.publication.lock().unwrap();
    for index in 0..MAX_PENDING_RECORDS + 3 {
        context
            .0
            .emit(json!({"event": "queued_probe", "index": index}));
    }
    drop(publisher);
    teardown(scheduler, context);
    drop(guard);
    let records = records(&capture);
    assert_eq!(
        records
            .iter()
            .filter(|r| r["event"] == "queued_probe")
            .count(),
        MAX_PENDING_RECORDS
    );
    let loss = records
        .iter()
        .find(|r| r["event"] == "observation_loss")
        .unwrap();
    assert_eq!(loss["reason"], "publication_queue_overflow");
    assert_eq!(loss["dropped_count"], 3);
    assert_eq!(records.last().unwrap()["event"], "trace_finished");
    assert_eq!(records.last().unwrap()["lost_observations"], 3);
    assert_eq!(records.last().unwrap()["evidence_valid"], false);
}

#[test]
fn execution_leases_share_cache_publication_order_and_close_after_late_results() {
    use ts_control::execution_observation::{ExecutionResult, ForwardLease, RequestObservation};
    let mut scheduler = scheduler();
    let (writer, guard, capture) = writer();
    let context = install(&mut scheduler, writer);
    let sink = context.execution_sink();
    let owner = RequestObservation::new(sink.clone(), "request".into(), 77, false);
    context
        .0
        .emit(json!({"event": "restore_completed", "admission_id": 77,
        "request_id": "request", "slot": 5, "scope": "local_physical"}));
    let mut pending = ForwardLease::default();
    pending.push(owner.clone(), 5, true, 8);
    owner.terminal("length".into());
    drop(owner);
    pending.complete(
        0,
        Some(&ExecutionResult {
            admission_ids: vec![77],
            path: "decode_graph".into(),
            verify_width: 8,
            live_requests: 1,
            request_capacity: 4,
            query_capacity: 32,
            graph_segments: 1,
            verification: vec![[6, 0]],
        }),
        0,
        0,
        false,
        true,
    );
    drop(pending);
    drop(sink);
    teardown(scheduler, context);
    drop(guard);
    let bytes = capture.0.lock().unwrap();
    let records: Vec<Value> = std::str::from_utf8(&bytes)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line.split_once("] ").unwrap().1).unwrap())
        .collect();
    let index = |event| records.iter().position(|r| r["event"] == event).unwrap();
    assert!(index("restore_completed") < index("target_round_dispatched"));
    assert!(index("target_round_dispatched") < index("target_round_completed"));
    assert!(index("target_round_completed") < index("request_closed"));
    for (order, record) in records.iter().enumerate() {
        assert_eq!(record["publication_order"], order);
    }
    let execution: Vec<_> = records
        .iter()
        .filter(|r| r["protocol"] == ts_control::execution_observation::PROTOCOL)
        .collect();
    for (sequence, record) in execution.iter().enumerate() {
        assert_eq!(record["record_sequence"], sequence);
    }
    assert_eq!(execution.last().unwrap()["event"], "trace_finished");
    assert_eq!(execution.last().unwrap()["evidence_valid"], true);
}
