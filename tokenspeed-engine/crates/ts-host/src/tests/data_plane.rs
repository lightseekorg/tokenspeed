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
#[path = "data_plane/result_buffers.rs"]
mod result_buffers;

/// Start the interpreter once per test process, with the scheduler
/// module built in and the repository's `tokenspeed_scheduler` package
/// importable.
fn interpreter() {
    static START: Once = Once::new();
    START.call_once(|| {
        // The test binary is not in a Python prefix; point at the one the
        // crate was built against.
        std::env::set_var("PYTHONHOME", env!("TS_HOST_TEST_PYTHONHOME"));
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

    def check_memory_transition(self, tags, release):
        self.note("check_memory_transition")
        if release and tags == ["weights"]:
            return (False, "release KV before weights")
        return (True, "")

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

fn start(plane: &Py<PyAny>) -> PythonDataPlane {
    PythonDataPlane::start(Python::attach(|py| plane.clone_ref(py)), None).unwrap()
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
fn a_memory_preflight_refusal_allows_another_check() {
    let (plane, _) = plane();
    let mut data_plane = start(&plane);
    let tags = ["weights".to_owned()];
    assert_eq!(
        data_plane.check_memory_transition(&tags, true).unwrap(),
        (false, "release KV before weights".to_owned())
    );
    assert_eq!(
        data_plane.check_memory_transition(&tags, false).unwrap(),
        (true, String::new())
    );

    let checks: Vec<_> = calls(&plane)
        .into_iter()
        .filter(|(name, _)| name == "check_memory_transition")
        .collect();
    assert_eq!(checks.len(), 2);
    assert_eq!(checks[0].1, checks[1].1);
    assert_ne!(checks[0].1, this_thread());
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
            let mut data_plane = PythonDataPlane::start(plane, None).unwrap();
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
