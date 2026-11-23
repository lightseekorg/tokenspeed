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

//! The two-thread contract, against a Python stand-in for `HostDataPlane`
//! that records which thread made each call.

use std::{ffi::CString, sync::Once, time::Duration};

use pyo3::types::PyDict;
use ts_scheduler::{ForwardBatchView, Scheduler};

use super::*;

#[path = "data_plane/cpu_benchmark.rs"]
mod cpu_benchmark;
#[path = "data_plane/pd_notice.rs"]
mod pd_notice;
#[path = "data_plane/result_buffers.rs"]
mod result_buffers;
// The control crate's owner for tests, which says what a test scripts.
#[allow(dead_code)]
#[path = "../../../ts-control/src/shared_l2/testing/owner.rs"]
mod shared_l2_owner;

/// Start the interpreter once per test process, with the scheduler
/// module loaded and the repository's `tokenspeed_scheduler` package
/// importable.
fn interpreter() {
    static START: Once = Once::new();
    START.call_once(|| {
        // The test binary is not in a Python prefix; point at the one the
        // crate was built against.
        std::env::set_var("PYTHONHOME", env!("TS_HOST_TEST_PYTHONHOME"));
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

#[test]
fn python_starts_and_stops_the_host_span_recording() {
    interpreter();
    Python::attach(|py| {
        let host_spans = Bound::new(py, crate::host_spans::HostSpans).unwrap();
        host_spans.call_method0("start").unwrap();
        drop(profile::span("recorded"));
        let (spans, threads): (Vec<crate::host_spans::PySpan>, Vec<(u64, String)>) =
            host_spans.call_method0("stop").unwrap().extract().unwrap();
        drop(profile::span("after_stop"));
        // Other tests share the process: find this test's spans by name.
        let (_, category, tid, begin_ns, end_ns) = spans
            .iter()
            .find(|(name, ..)| name == "recorded")
            .expect("the span recorded between start and stop")
            .clone();
        assert_eq!(category, profile::HOST);
        assert!(0 < begin_ns && begin_ns <= end_ns);
        assert!(threads.iter().any(|(thread, _)| *thread == tid));
        assert!(!profile::recording());
    });
}

/// The stand-in, and an idle scheduler to draw plans from. A launch
/// returns the next token, from 100; the launch numbered `held` waits
/// for `unblock`, and every launch raises once `failing` is set.
const PLANE: &str = r#"
import threading

import tokenspeed_scheduler as ts


class Guard:
    def __init__(self, plane):
        self.plane = plane

    def __enter__(self):
        self.plane.note("guard enter")

    def __exit__(self, *exc):
        self.plane.note("guard exit")


class Plane:
    def __init__(self):
        self.calls = []
        self.admissions = []
        self.next_token = 100
        self.held = None
        self.unblock = threading.Event()
        self.failing = False
        self.completion_failing = False
        self.cumulative_probs = None

    def note(self, name):
        self.calls.append((name, threading.get_ident()))

    def control_thread_guard(self):
        return Guard(self)

    def enter_forward_thread(self):
        self.note("enter_forward_thread")

    def exit_forward_thread(self):
        self.note("exit_forward_thread")

    def admit(self, rows):
        self.admissions.extend(rows)

    def execute(self, plan, dp, ngram, grammar_advance, logprobs, history):
        self.note("execute")
        token = self.next_token
        self.next_token += 1
        if token == self.held:
            self.unblock.wait()
        if self.failing:
            raise RuntimeError("the launch failed")
        return token

    def complete(self, launched):
        self.note("complete")
        if self.completion_failing:
            raise RuntimeError("the completion failed")
        from array import array
        cumulative = None if self.cumulative_probs is None else array('d', self.cumulative_probs)
        return ([launched], [1], None, None, None, None, None, None, cumulative)

    def note_committed(self, launched):
        self.note("note_committed")

    def release(self, rids):
        self.note("release")

    def release_memory(self, tags):
        self.note("release_memory")
        raise RuntimeError("the memory release failed")

    def poll_cache_events(self):
        self.note("poll_cache_events")
        return [("WriteBackDoneEvent", 3)]


cfg = ts.SchedulerConfig()
cfg.prefix_granularity = 16
cfg.max_scheduled_tokens = 64
cfg.max_batch_size = 4
cfg.max_resident_requests = 4
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
result = (Plane(), ts.Scheduler(cfg))
"#;

fn plane() -> (Py<PyAny>, Scheduler) {
    interpreter();
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(&CString::new(PLANE).unwrap(), Some(&globals), None)
            .unwrap_or_else(|error| panic!("{}", describe(py, &error)));
        let (plane, scheduler): (Py<PyAny>, Bound<'_, PyAny>) = globals
            .get_item("result")
            .unwrap()
            .unwrap()
            .extract()
            .unwrap();
        (plane, Scheduler::from_python(&scheduler).unwrap())
    })
}

/// The data plane of a rank that joined no shared L2 pool.
fn start(plane: &Py<PyAny>) -> PythonDataPlane {
    let (plane, shared_l2_handle) = Python::attach(|py| {
        let shared_l2_handle = Py::new(py, crate::pool::Handle::new()).unwrap();
        (plane.clone_ref(py), shared_l2_handle)
    });
    PythonDataPlane::start(plane, None, shared_l2_handle).unwrap()
}

/// A one-row batch: the stand-in launches whatever it is handed.
fn batch() -> ForwardBatchView {
    ForwardBatchView {
        request_ids: vec!["r".to_owned()],
        request_pool_indices: vec![0],
        input_lengths: vec![1],
        prefill_lengths: vec![1],
        extend_prefix_lens: Vec::new(),
        extend_replay_lens: Vec::new(),
        num_extends: 0,
    }
}

fn set(plane: &Py<PyAny>, name: &str, value: i64) {
    Python::attach(|py| plane.setattr(py, name, value).unwrap());
}

/// The calls the stand-in recorded, with the thread each ran on.
fn calls(plane: &Py<PyAny>) -> Vec<(String, u64)> {
    Python::attach(|py| plane.getattr(py, "calls").unwrap().extract(py).unwrap())
}

fn this_thread() -> u64 {
    Python::attach(|py| {
        py.import("threading")
            .unwrap()
            .call_method0("get_ident")
            .unwrap()
            .extract()
            .unwrap()
    })
}

#[test]
fn admission_identity_and_stop_criteria_reach_python_together() {
    let (plane, _) = plane();
    let mut data_plane = start(&plane);
    let mut state = ts_control::request::RequestState::new(
        vec![1, 2, 3],
        8,
        std::collections::BTreeSet::from([9]),
        true,
        None,
        false,
    );
    state.admission = 42;
    let request = Admission {
        rid: "reused".to_owned(),
        spec: ts_scheduler::RequestSpec {
            request_id: "reused".to_owned(),
            session_id: String::new(),
            cache_group_id: String::new(),
            shared_prefix_tokens: Vec::new(),
            peer_holder: String::new(),
            tokens: vec![1, 2, 3],
            max_new_tokens: 8,
            max_cached_prefix_tokens: i32::MAX,
        },
        state,
        sampling_params: ts_wire::SamplingParams::default(),
        bootstrap: None,
        multimodal: None,
        embeddings: Vec::new(),
        grammar: None,
    };
    data_plane.admit(&[request]).unwrap();
    data_plane.ask(|reply| Op::Barrier { reply }).unwrap();
    Python::attach(|py| {
        let rows = plane.bind(py).getattr("admissions").unwrap();
        let row = rows.get_item(0).unwrap();
        assert_eq!(
            row.get_item(0).unwrap().extract::<String>().unwrap(),
            "reused"
        );
        assert_eq!(row.get_item(1).unwrap().extract::<u64>().unwrap(), 42);
        assert_eq!(row.get_item(3).unwrap().extract::<Vec<i32>>().unwrap(), [9]);
        assert_eq!(row.get_item(4).unwrap().extract::<i64>().unwrap(), 11);
    });
}

#[test]
fn completion_keeps_the_selected_tokens_cumulative_probabilities() {
    let (plane, mut scheduler) = plane();
    Python::attach(|py| plane.setattr(py, "cumulative_probs", vec![0.875]).unwrap());
    let batch = batch();
    let dispatch = Dispatch {
        forward: Some(&batch),
        ..Dispatch::default()
    };
    let mut data_plane = start(&plane);
    let forward = data_plane
        .execute(scheduler.next_execution_plan(true, false), &dispatch)
        .unwrap()
        .unwrap();
    let result = data_plane.complete(forward).unwrap();
    assert_eq!(result.output_tokens, [100]);
    assert_eq!(result.output_token_cumulative_probs, Some(vec![0.875]));
}

#[test]
fn the_forward_thread_runs_the_queue_and_the_control_thread_its_queries() {
    let (plane, mut scheduler) = plane();
    let batch = batch();
    let dispatch = Dispatch {
        forward: Some(&batch),
        ..Dispatch::default()
    };
    let mut data_plane = start(&plane);
    let forward = data_plane
        .execute(scheduler.next_execution_plan(true, false), &dispatch)
        .unwrap()
        .unwrap();
    assert_eq!(data_plane.complete(forward).unwrap().output_tokens, [100]);
    let polled = data_plane.poll_cache_events().unwrap();
    assert_eq!(polled.len(), 1);
    data_plane.release(&["r".to_owned()]).unwrap();
    // Answered once everything queued before it ran.
    data_plane.ask(|reply| Op::Barrier { reply }).unwrap();
    drop(data_plane);

    let control = this_thread();
    let calls = calls(&plane);
    let on = |thread: u64| -> Vec<&str> {
        calls
            .iter()
            .filter(|(_, ran_on)| *ran_on == thread)
            .map(|(name, _)| name.as_str())
            .collect()
    };
    assert_eq!(
        on(control),
        ["guard enter", "complete", "poll_cache_events", "guard exit"]
    );
    let forward_thread = calls
        .iter()
        .find(|(name, _)| name == "enter_forward_thread")
        .unwrap()
        .1;
    assert_ne!(forward_thread, control);
    assert_eq!(
        on(forward_thread),
        [
            "enter_forward_thread",
            "execute",
            "note_committed",
            "release",
            "exit_forward_thread"
        ]
    );
}

#[test]
fn a_healthy_lifecycle_exits_without_notifying_the_launcher() {
    crate::tests::assert_no_launcher_notification(
        "data_plane::tests::a_healthy_lifecycle_exits_without_notifying_the_launcher",
        || {
            let launcher = LAUNCHER_FAILURE.get_or_init(crate::LauncherFailure::for_parent);
            let (plane, mut scheduler) = plane();
            let mut data_plane = start(&plane);
            let batch = batch();
            let dispatch = Dispatch {
                forward: Some(&batch),
                ..Dispatch::default()
            };
            let forward = data_plane
                .execute(scheduler.next_execution_plan(true, false), &dispatch)
                .unwrap()
                .unwrap();
            assert_eq!(data_plane.complete(forward).unwrap().output_tokens, [100]);
            data_plane.ask(|reply| Op::Barrier { reply }).unwrap();
            let mut closed = 0;
            let result = crate::finish_serving(Ok(()), launcher, || {
                drop(data_plane);
                let recorded = calls(&plane);
                assert_eq!(
                    recorded
                        .iter()
                        .filter(|(name, _)| name == "exit_forward_thread")
                        .count(),
                    1
                );
                assert_eq!(
                    recorded
                        .iter()
                        .filter(|(name, _)| name == "guard exit")
                        .count(),
                    1
                );
                closed += 1;
                Ok(())
            });
            assert!(result.is_ok());
            assert_eq!(closed, 1);
        },
    );
}

#[test]
fn a_commit_does_not_wait_behind_a_later_launch() {
    let (plane, mut scheduler) = plane();
    let batch = batch();
    let dispatch = Dispatch {
        forward: Some(&batch),
        ..Dispatch::default()
    };
    set(&plane, "held", 101);
    let mut data_plane = start(&plane);
    let first = data_plane
        .execute(scheduler.next_execution_plan(true, false), &dispatch)
        .unwrap()
        .unwrap();
    // The forward thread will sit in this launch until released.
    let second = data_plane
        .execute(scheduler.next_execution_plan(true, false), &dispatch)
        .unwrap()
        .unwrap();
    // A commit queued behind it would wait forever; release the held
    // launch after a while regardless, and fail the test then.
    let (cancel, cancelled) = mpsc::channel::<()>();
    let held = Python::attach(|py| plane.clone_ref(py));
    let watchdog = thread::spawn(move || {
        let timed_out = cancelled.recv_timeout(Duration::from_secs(10)).is_err();
        Python::attach(|py| held.getattr(py, "unblock")?.call_method0(py, "set")).unwrap();
        timed_out
    });
    assert_eq!(data_plane.complete(first).unwrap().output_tokens, [100]);
    cancel.send(()).unwrap();
    assert!(
        !watchdog.join().unwrap(),
        "the commit waited for the launch"
    );
    assert_eq!(data_plane.complete(second).unwrap().output_tokens, [101]);
}

#[test]
fn a_failed_launch_fails_the_commit_waiting_for_it() {
    let (plane, mut scheduler) = plane();
    let batch = batch();
    let dispatch = Dispatch {
        forward: Some(&batch),
        ..Dispatch::default()
    };
    Python::attach(|py| plane.setattr(py, "failing", true).unwrap());
    let mut data_plane = start(&plane);
    let forward = data_plane
        .execute(scheduler.next_execution_plan(true, false), &dispatch)
        .unwrap()
        .unwrap();
    let error = data_plane.complete(forward).unwrap_err();
    assert!(error.0.contains("the launch failed"), "{}", error.0);
    // Every later call reports the failure, on either thread's side.
    assert!(data_plane.poll_cache_events().is_err());
    assert!(data_plane.release(&["r".to_owned()]).is_err());
}

#[test]
fn a_failed_completion_notifies_the_launcher_before_logging() {
    crate::tests::assert_launcher_notified_before_blocking(
        "data_plane::tests::a_failed_completion_notifies_the_launcher_before_logging",
        || {
            let (plane, mut scheduler) = plane();
            Python::attach(|py| plane.setattr(py, "completion_failing", true).unwrap());
            let batch = batch();
            let dispatch = Dispatch {
                forward: Some(&batch),
                ..Dispatch::default()
            };
            LAUNCHER_FAILURE.get_or_init(crate::LauncherFailure::for_parent);
            let mut data_plane = start(&plane);
            let forward = data_plane
                .execute(scheduler.next_execution_plan(true, false), &dispatch)
                .unwrap()
                .unwrap();
            let error = data_plane.complete(forward).unwrap_err();
            assert!(error.0.contains("the completion failed"), "{}", error.0);
        },
    );
}

#[test]
fn distinct_failures_are_logged_without_replacing_the_original() {
    let shared = Shared::default();
    let (_, output) = crate::tests::capture_logs(|| {
        shared.fail("first launch failure".to_owned());
        shared.fail("first launch failure".to_owned());
        shared.fail("second completion failure".to_owned());
    });

    assert_eq!(shared.failure().as_deref(), Some("first launch failure"));
    assert_eq!(
        output.matches("first launch failure").count(),
        1,
        "{output}"
    );
    assert_eq!(
        output.matches("second completion failure").count(),
        1,
        "{output}"
    );
}

#[test]
fn a_failed_answer_is_logged_once_across_both_threads() {
    crate::tests::run_in_child(
        "data_plane::tests::a_failed_answer_is_logged_once_across_both_threads",
        || {
            let launcher = LAUNCHER_FAILURE.get_or_init(|| crate::LauncherFailure {
                notification: std::sync::OnceLock::new(),
                logged: Default::default(),
                parent: None,
            });
            let (result, output) = crate::tests::capture_all_thread_logs(|| {
                let (plane, _scheduler) = plane();
                let mut data_plane = start(&plane);
                let error = data_plane
                    .release_memory(&["weights".to_owned()])
                    .unwrap_err();
                assert!(error.0.contains("the memory release failed"), "{}", error.0);
                assert!(
                    data_plane
                        .shared
                        .failure()
                        .is_some_and(|failure| failure.contains("the memory release failed")),
                    "a failed answer must publish the data-plane failure before returning"
                );
                crate::finish_serving(Err(crate::LoopError::DataPlane(error)), launcher, || {
                    drop(data_plane);
                    Ok(())
                })
            });
            assert!(matches!(result, Err(crate::HostError::Loop(_))));
            assert_eq!(
                output
                    .matches("RuntimeError: the memory release failed")
                    .count(),
                1,
                "{output}"
            );
        },
    );
}

#[test]
fn a_blocked_error_log_does_not_hide_the_failure_or_block_later_errors() {
    crate::tests::run_in_child(
        "data_plane::tests::a_blocked_error_log_does_not_hide_the_failure_or_block_later_errors",
        || {
            assert_failure_is_published_before_logging(
                crate::LauncherFailure {
                    notification: std::sync::OnceLock::new(),
                    logged: Default::default(),
                    parent: None,
                },
                "first launch failure",
                0,
            );
        },
    );
}

#[cfg(target_os = "linux")]
#[test]
fn a_blocked_signal_error_log_does_not_delay_waking_waiters() {
    crate::tests::run_in_child(
        "data_plane::tests::a_blocked_signal_error_log_does_not_delay_waking_waiters",
        || {
            deny_kill_syscall();
            assert_failure_is_published_before_logging(
                crate::LauncherFailure::for_parent(),
                "could not notify the launcher",
                1,
            );
        },
    );
}

#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
fn deny_kill_syscall() {
    // In the isolated test child, make the real kill syscall fail without
    // risking a signal to the test runner. New threads inherit this filter.
    // SAFETY: These helpers construct BPF instructions with a valid load
    // of the syscall number and an in-bounds jump.
    let mut filter = unsafe {
        [
            libc::BPF_STMT((libc::BPF_LD | libc::BPF_W | libc::BPF_ABS) as u16, 0),
            libc::BPF_JUMP(
                (libc::BPF_JMP | libc::BPF_JEQ | libc::BPF_K) as u16,
                libc::SYS_kill as u32,
                0,
                1,
            ),
            libc::BPF_STMT(
                (libc::BPF_RET | libc::BPF_K) as u16,
                libc::SECCOMP_RET_ERRNO | libc::EPERM as u32,
            ),
            libc::BPF_STMT(
                (libc::BPF_RET | libc::BPF_K) as u16,
                libc::SECCOMP_RET_ALLOW,
            ),
        ]
    };
    let program = libc::sock_fprog {
        len: filter.len() as u16,
        filter: filter.as_mut_ptr(),
    };
    let zero: libc::c_ulong = 0;
    // SAFETY: The filter buffer and its descriptor live through prctl, which
    // copies the program. The unused variadic arguments are zero.
    unsafe {
        assert_eq!(
            libc::prctl(
                libc::PR_SET_NO_NEW_PRIVS,
                1 as libc::c_ulong,
                zero,
                zero,
                zero
            ),
            0,
            "could not enable no_new_privs: {}",
            std::io::Error::last_os_error()
        );
        assert_eq!(
            libc::prctl(
                libc::PR_SET_SECCOMP,
                libc::SECCOMP_MODE_FILTER as libc::c_ulong,
                std::ptr::from_ref(&program) as libc::c_ulong,
                zero,
                zero,
            ),
            0,
            "could not install seccomp filter: {}",
            std::io::Error::last_os_error()
        );
    }
}

fn assert_failure_is_published_before_logging(
    launcher: crate::LauncherFailure,
    first_log: &str,
    signal_errors: usize,
) {
    use std::{
        io::{self, Write},
        sync::atomic::{AtomicBool, Ordering},
    };

    assert!(LAUNCHER_FAILURE.set(launcher).is_ok());

    #[derive(Clone)]
    struct HeldFirstWrite {
        entered: mpsc::Sender<String>,
        release: Arc<Mutex<mpsc::Receiver<()>>>,
        blocked: Arc<AtomicBool>,
        output: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for HeldFirstWrite {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if !self.blocked.swap(true, Ordering::Relaxed) {
                self.entered
                    .send(String::from_utf8_lossy(bytes).into_owned())
                    .map_err(io::Error::other)?;
                self.release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .map_err(io::Error::other)?;
            }
            self.output.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let (entered, writing) = mpsc::channel();
    let (release, held) = mpsc::channel();
    let output = Arc::new(Mutex::new(Vec::new()));
    let writer = HeldFirstWrite {
        entered,
        release: Arc::new(Mutex::new(held)),
        blocked: Arc::new(AtomicBool::new(false)),
        output: output.clone(),
    };
    let subscriber = tracing_subscriber::fmt()
        .with_ansi(false)
        .without_time()
        .with_writer(move || writer.clone())
        .finish();
    let dispatch = tracing::Dispatch::new(subscriber);
    let shared = Arc::new(Shared::default());
    let (waiting, ready) = mpsc::channel();
    let (report_later, later_allowed) = mpsc::channel();
    let (done, finished) = mpsc::channel();
    let observing = {
        let shared = shared.clone();
        let dispatch = dispatch.clone();
        thread::spawn(move || {
            let state = shared.state();
            // Hold the state lock until entering the wait, so the reporter
            // cannot publish the failure before this thread starts waiting.
            waiting.send(()).unwrap();
            let state = shared
                .changed
                .wait_while(state, |state| state.failure.is_none())
                .unwrap();
            let first = state.failure.clone().unwrap();
            drop(state);
            later_allowed.recv_timeout(Duration::from_secs(5)).unwrap();
            tracing::dispatcher::with_default(&dispatch, || {
                shared.fail("first launch failure".to_owned());
                shared.fail("second completion failure".to_owned());
            });
            done.send(first).unwrap();
        })
    };
    ready.recv_timeout(Duration::from_secs(5)).unwrap();

    let reporting = {
        let shared = shared.clone();
        thread::spawn(move || {
            tracing::dispatcher::with_default(&dispatch, || {
                shared.fail("first launch failure".to_owned());
            });
        })
    };
    let held_log = writing.recv_timeout(Duration::from_secs(5)).unwrap();
    report_later.send(()).unwrap();
    // Waking the waiter and reporting a different error must both work
    // while the first log is still blocked.
    let observed_before_release = finished.recv_timeout(Duration::from_secs(1));
    release.send(()).unwrap();
    reporting.join().unwrap();
    assert!(held_log.contains(first_log), "{held_log}");
    assert_eq!(
        observed_before_release.expect("failure publication or wake waited for the log writer"),
        "first launch failure"
    );
    observing.join().unwrap();
    assert_eq!(shared.failure().as_deref(), Some("first launch failure"));
    let output = String::from_utf8(output.lock().unwrap().clone()).unwrap();
    assert_eq!(
        output.matches("first launch failure").count(),
        1,
        "{output}"
    );
    assert_eq!(
        output.matches("second completion failure").count(),
        1,
        "{output}"
    );
    assert_eq!(
        output.matches("could not notify the launcher").count(),
        signal_errors,
        "{output}"
    );
}

/// What a rank that joined a pool hands the host (`SharedL2ClientSettings`
/// in `cache/shared_l2/rank.py`), and the forward thread's and the
/// completion query's calls of the shared L2 handle.
const POOL_RANK: &str = r#"
class Fields:
    def __init__(self, **fields):
        for name, value in fields.items():
            setattr(self, name, value)


def settings(owner_address, signal_words_address):
    return Fields(
        owner_address=owner_address,
        client_id=7,
        request_timeout_ms=5000,
        load_workers=2,
        write_workers=1,
        lcm_bytes=64,
        providers=[
            Fields(session="node0", base=10000, lcm_blocks=3, start=0),
            Fields(session="node1", base=20000, lcm_blocks=5, start=3),
        ],
        groups=[
            # Two KV pages in an LCM block, k then v; a State page fills one.
            Fields(
                packing=2,
                page_bytes=32,
                fields=[
                    Fields(
                        device_address=0,
                        device_stride=32,
                        bytes=12,
                        page_offset=0,
                        layer=0,
                    ),
                    Fields(
                        device_address=16,
                        device_stride=32,
                        bytes=16,
                        page_offset=12,
                        layer=1,
                    ),
                ],
            ),
            Fields(
                packing=1,
                page_bytes=64,
                fields=[
                    Fields(
                        device_address=2000,
                        device_stride=64,
                        bytes=64,
                        page_offset=0,
                        layer=1,
                    )
                ],
            ),
        ],
        num_layers=2,
        signal_words_address=signal_words_address,
        num_signal_words=6,
    )


def load(handle):
    # KV page 7 (node1's first) into Device block 5, State page 2 into 3.
    handle.submit_load(
        op_ids=[41, 42], rows=[(0, 100, 5), (1, 101, 3)], start_word=3, seq=9
    )


def unmapped(handle):
    handle.submit_load(op_ids=[43], rows=[(0, 999, 5)], start_word=0, seq=1)


def outside(handle):
    # Device block 2000 lies outside the test memory.
    handle.submit_load(op_ids=[44], rows=[(0, 100, 2000)], start_word=0, seq=1)
"#;

#[test]
fn a_pool_ranks_loads_run_and_are_acknowledged_in_the_host() {
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    };

    use ts_control::peer::read::Arena;
    use ts_scheduler::{CacheEvent, CacheEventKind, SharedL2PageView};

    // The rank's Python data plane: it acknowledges a write-back whenever
    // it is asked for cache acknowledgments.
    let (plane, _) = plane();
    // An owner that says `ok` to every request.
    let owner = shared_l2_owner::TestOwner::serve(|_| Some(shared_l2_owner::ok(Vec::new())));
    let address = owner.address.clone();
    // The rank's signal words: two sets of a start word and two layers.
    let words: Vec<AtomicU32> = (0..6).map(|_| AtomicU32::new(0)).collect();
    let (handle, globals) = Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(&CString::new(POOL_RANK).unwrap(), Some(&globals), None)
            .unwrap_or_else(|error| panic!("{}", describe(py, &error)));
        let handle = Bound::new(py, crate::pool::Handle::new()).unwrap();
        (handle.unbind(), globals.unbind())
    });
    let mut data_plane = {
        let (plane, handle) = Python::attach(|py| (plane.clone_ref(py), handle.clone_ref(py)));
        PythonDataPlane::start(plane, None, handle).unwrap()
    };
    let call = |function: &str| {
        Python::attach(|py| {
            globals
                .bind(py)
                .get_item(function)
                .unwrap()
                .unwrap()
                .call1((handle.clone_ref(py),))
                .map(drop)
                .map_err(|error| describe(py, &error))
        })
    };

    // A rank that started no client has no loads, no client for the loop
    // and nothing to close.
    assert!(call("load").unwrap_err().contains("has not started"));
    assert!(handle.get().take_client().is_none());
    assert!(Python::attach(|py| handle
        .call_method0(py, "close")
        .unwrap()
        .extract::<bool>(py)
        .unwrap()));

    // The settings the rank's Python hands over become the pool's geometry
    // and the signal words' location.
    let (settings, signal_words) = Python::attach(|py| {
        let settings = globals
            .bind(py)
            .get_item("settings")
            .unwrap()
            .unwrap()
            .call1((&address, words.as_ptr() as usize))
            .unwrap();
        crate::pool::client_settings(&settings)
            .unwrap_or_else(|error| panic!("{}", describe(py, &error)))
    });
    assert_eq!(
        (settings.owner_address.as_str(), settings.client_id),
        (address.as_str(), 7)
    );
    assert_eq!(settings.request_timeout, Duration::from_secs(5));
    assert_eq!(
        (settings.load_workers.get(), settings.write_workers.get()),
        (2, 1)
    );
    assert_eq!(settings.geometry.num_layers, 2);
    let sessions: Vec<_> = settings
        .geometry
        .providers
        .iter()
        .map(|provider| provider.session.as_str())
        .collect();
    assert_eq!(sessions, ["node0", "node1"]);
    // The pool ends at KV page 16 and State page 8, on node1.
    assert_eq!(settings.geometry.page_address(0, 0, 16).unwrap().0, 1);
    assert_eq!(settings.geometry.page_address(1, 0, 8).unwrap().0, 1);
    assert!(settings.geometry.page_address(1, 0, 9).is_err());

    // Test memory stands in for the Device cache and the providers.
    let arena = Arena::new(30_000);
    let (kv_page, state_page) = (20_000, 10_000 + 64);
    arena.write(kv_page, &[1; 12]);
    arena.write(kv_page + 12, &[2; 16]);
    arena.write(state_page, &[3; 64]);
    handle
        .get()
        .start_client_with(
            settings,
            Arc::new(arena.clone()),
            Arc::new(arena.clone()),
            Arc::new(signal_words),
        )
        .unwrap();
    let mut client = handle.get().take_client().expect("the rank's client");
    assert!(handle.get().take_client().is_none());

    // The forward thread hands a load over; a proxy block that stands for
    // no page is refused there.
    client
        .map_proxy_pages(&[
            SharedL2PageView {
                group_id: 0,
                block: 100,
                bucket: 0,
                page: 7,
            },
            SharedL2PageView {
                group_id: 1,
                block: 101,
                bucket: 0,
                page: 2,
            },
        ])
        .unwrap();
    assert!(call("unmapped")
        .unwrap_err()
        .contains("stands for no pool page"));
    call("load").unwrap();
    // Nothing is read before the start word of the second word set lands.
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(data_plane.poll_cache_events().unwrap(), []);
    words[3].store(9, Ordering::Relaxed);
    // The data plane's cache poll acknowledges the load's operations from
    // the loader's finished loads, and never asks Python.
    let started = std::time::Instant::now();
    let mut acknowledged = Vec::new();
    while acknowledged.is_empty() {
        assert!(started.elapsed() < Duration::from_secs(10));
        acknowledged = data_plane.poll_cache_events().unwrap();
        std::thread::sleep(Duration::from_millis(1));
    }
    let load_back_done = |op_id| CacheEvent {
        kind: CacheEventKind::LoadBackDone,
        op_id,
    };
    assert_eq!(acknowledged, [load_back_done(41), load_back_done(42)]);
    assert!(calls(&plane)
        .iter()
        .all(|(name, _)| name != "poll_cache_events"));
    // Each layer's ready word holds the load's value, set through the
    // words' address, and the blocks hold the pages.
    assert_eq!(
        words
            .iter()
            .map(|word| word.load(Ordering::Relaxed))
            .collect::<Vec<_>>(),
        [0, 0, 0, 9, 9, 9]
    );
    assert_eq!(arena.read_bytes(5 * 32, 12), [1; 12]);
    assert_eq!(arena.read_bytes(16 + 5 * 32, 16), [2; 16]);
    assert_eq!(arena.read_bytes(2000 + 3 * 64, 64), [3; 64]);

    // A load that fails is the rank's failure: the cache poll raises it in
    // the pool's own words, and the client has it for the loop's check.
    call("outside").unwrap();
    words[0].store(1, Ordering::Relaxed);
    let started = std::time::Instant::now();
    let failure = loop {
        match data_plane.poll_cache_events() {
            Err(failure) => break failure.0,
            Ok(acknowledged) => assert_eq!(acknowledged, []),
        }
        assert!(started.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(
        failure,
        "shared L2 read from node1 failed: a span outside the arena"
    );
    assert_eq!(client.check_failure().unwrap_err().0, failure);

    // The rank's shutdown closes the shared L2 handle: every thread ended.
    assert!(Python::attach(|py| handle
        .call_method0(py, "close")
        .unwrap()
        .extract::<bool>(py)
        .unwrap()));
    assert!(call("load").unwrap_err().contains("have stopped"));
}
