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
        return ([launched], [1], None, None, None, None, None, None, self.cumulative_probs)

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
            LAUNCHER_FAILURE.get_or_init(|| crate::LauncherFailure {
                notification: std::sync::OnceLock::new(),
                logged: Default::default(),
                parent: None,
            });
            use std::{
                io::{self, Write},
                sync::atomic::{AtomicBool, Ordering},
            };

            #[derive(Clone)]
            struct HeldFirstWrite {
                entered: mpsc::Sender<()>,
                release: Arc<Mutex<mpsc::Receiver<()>>>,
                blocked: Arc<AtomicBool>,
                output: Arc<Mutex<Vec<u8>>>,
            }

            impl Write for HeldFirstWrite {
                fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                    if !self.blocked.swap(true, Ordering::Relaxed) {
                        self.entered.send(()).map_err(io::Error::other)?;
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
            let reporting = {
                let shared = shared.clone();
                let dispatch = dispatch.clone();
                thread::spawn(move || {
                    tracing::dispatcher::with_default(&dispatch, || {
                        shared.fail("first launch failure".to_owned());
                    });
                })
            };
            writing.recv_timeout(Duration::from_secs(5)).unwrap();

            let (done, finished) = mpsc::channel();
            let observing = {
                let shared = shared.clone();
                thread::spawn(move || {
                    tracing::dispatcher::with_default(&dispatch, || {
                        let first = shared.take_launched(1).unwrap_err();
                        shared.fail("first launch failure".to_owned());
                        shared.fail("second completion failure".to_owned());
                        done.send(first).unwrap();
                    });
                })
            };
            // Neither observing the failure nor reporting a different one may wait
            // for the first thread's synchronous output stream.
            let observed_before_release = finished.recv_timeout(Duration::from_secs(1));
            release.send(()).unwrap();
            reporting.join().unwrap();
            if observed_before_release.is_err() {
                finished
                    .recv_timeout(Duration::from_secs(5))
                    .expect("observer did not finish after the log writer was released");
            }
            observing.join().unwrap();

            assert_eq!(
                observed_before_release.expect("failure publication waited for the log writer"),
                "first launch failure"
            );
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
        },
    );
}
