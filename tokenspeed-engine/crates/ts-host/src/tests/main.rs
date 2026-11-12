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
    io::{self, Read, Write},
    process::{Child, Command, ExitStatus, Stdio},
    sync::{Arc, Mutex, OnceLock},
    thread,
    time::{Duration, Instant},
};

use tokio::signal::unix::{signal, SignalKind};

use super::{finish_serving, parse_args, HostError, LauncherFailure, LoopError};

const CHILD: &str = "TS_HOST_SIGNAL_TEST_CHILD";
const SIGNAL_TIMEOUT: Duration = Duration::from_secs(5);
const CHILD_TIMEOUT: Duration = Duration::from_secs(20);
// Signal listeners belong to the test process, not to a test thread.
static SIGNAL_TEST: Mutex<()> = Mutex::new(());

/// Kill and reap a test child on failure instead of leaving a stuck rank in CI.
struct TestChild(Child);

impl TestChild {
    fn wait(&mut self) -> ExitStatus {
        let deadline = Instant::now() + CHILD_TIMEOUT;
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "test child did not exit");
            thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for TestChild {
    fn drop(&mut self) {
        if matches!(self.0.try_wait(), Ok(Some(_))) {
            return;
        }
        let _ = self.0.kill();
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match self.0.try_wait() {
                Ok(Some(_)) | Err(_) => return,
                Ok(None) => thread::sleep(Duration::from_millis(5)),
            }
        }
    }
}

fn test_child(test: &str) -> TestChild {
    TestChild(
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test])
            .env(CHILD, test)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap(),
    )
}

pub(super) fn run_in_child(test: &str, child: impl FnOnce()) {
    if std::env::var(CHILD).as_deref() == Ok(test) {
        child();
        return;
    }
    assert!(test_child(test).wait().success());
}

#[derive(Clone)]
struct LogWriter(Arc<Mutex<Vec<u8>>>);

impl Write for LogWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn capture_logs<T>(call: impl FnOnce() -> T) -> (T, String) {
    let output = Arc::new(Mutex::new(Vec::new()));
    let writer = LogWriter(output.clone());
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    let result = tracing::subscriber::with_default(subscriber, call);
    let logs = String::from_utf8(output.lock().unwrap().clone()).unwrap();
    (result, logs)
}

/// Capture production worker threads too; only call in an isolated test child.
pub(super) fn capture_all_thread_logs<T>(call: impl FnOnce() -> T) -> (T, String) {
    let output = Arc::new(Mutex::new(Vec::new()));
    let writer = LogWriter(output.clone());
    let subscriber = tracing_subscriber::fmt()
        .without_time()
        .with_ansi(false)
        .with_writer(move || writer.clone())
        .finish();
    tracing::subscriber::set_global_default(subscriber).unwrap();
    let result = call();
    let logs = String::from_utf8(output.lock().unwrap().clone()).unwrap();
    (result, logs)
}

#[test]
fn serving_error_notifies_launcher_before_teardown() {
    assert_launcher_notified_before_blocking(
        "tests::serving_error_notifies_launcher_before_teardown",
        || {
            let launcher = LauncherFailure::for_parent();
            let result = finish_serving(
                Err(LoopError::Scheduler("serving failed".to_owned())),
                &launcher,
                || {
                    wait_for_launcher();
                    Err(HostError::Python("close failed".to_owned()))
                },
            );
            assert!(
                matches!(result, Err(HostError::Loop(LoopError::Scheduler(message)))
                if message == "serving failed")
            );
        },
    );
}

fn wait_for_launcher() {
    // The launcher holds this pipe open until it gets SIGUSR1.
    io::copy(&mut io::stdin().lock(), &mut io::sink()).unwrap();
}

pub(super) fn assert_launcher_notified_before_blocking(test: &str, child: impl FnOnce()) {
    if std::env::var(CHILD).as_deref() == Ok(test) {
        tracing_subscriber::fmt()
            .with_writer(|| {
                wait_for_launcher();
                io::sink()
            })
            .init();
        child();
        return;
    }

    let _signal_test = SIGNAL_TEST
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut failure = {
        let _entered = runtime.enter();
        signal(SignalKind::user_defined1()).unwrap()
    };
    let mut rank = test_child(test);
    let notified = runtime.block_on(async {
        matches!(
            tokio::time::timeout(SIGNAL_TIMEOUT, failure.recv()).await,
            Ok(Some(()))
        )
    });
    let running = rank.0.try_wait().unwrap().is_none();
    drop(rank.0.stdin.take());
    assert!(rank.wait().success());
    assert!(
        notified && running,
        "no launcher signal while the child was alive and the pipe was held open"
    );
}

#[test]
fn cleanup_failure_is_logged_without_replacing_the_serving_error() {
    // No owning launcher in this in-process diagnostic test.
    let launcher = LauncherFailure {
        notification: OnceLock::new(),
        logged: Default::default(),
        parent: None,
    };
    let (result, logs) = capture_logs(|| {
        finish_serving(
            Err(LoopError::Scheduler("original serving failure".to_owned())),
            &launcher,
            || Err(HostError::Python("distinct cleanup failure".to_owned())),
        )
    });
    assert!(
        matches!(result, Err(HostError::Loop(LoopError::Scheduler(message)))
        if message == "original serving failure")
    );
    assert_eq!(
        logs.matches("original serving failure").count(),
        1,
        "{logs}"
    );
    assert_eq!(
        logs.matches("distinct cleanup failure").count(),
        1,
        "{logs}"
    );
    assert!(logs.contains("rank cleanup also failed"), "{logs}");
}

#[test]
fn cleanup_failure_after_success_still_propagates() {
    let launcher = LauncherFailure {
        notification: OnceLock::new(),
        logged: Default::default(),
        parent: None,
    };
    let error = finish_serving(Ok(()), &launcher, || {
        Err(HostError::Python("cleanup failed".to_owned()))
    })
    .unwrap_err();
    let ((), logs) = capture_logs(|| launcher.report(&error));
    assert!(matches!(&error, HostError::Python(message) if message == "cleanup failed"));
    assert_eq!(logs.matches("cleanup failed").count(), 1, "{logs}");
}

#[test]
fn reports_keep_distinct_errors_without_repeating_propagated_failures() {
    use ts_control::event_loop::DataPlaneError;

    let launcher = LauncherFailure {
        notification: OnceLock::new(),
        logged: Default::default(),
        parent: None,
    };
    let ((), logs) = capture_logs(|| {
        let original = DataPlaneError("original failure".to_owned());
        launcher.report(&original);
        launcher.report(HostError::Loop(LoopError::DataPlane(original)));
        launcher.report(LoopError::Scheduler("distinct control failure".to_owned()));
        launcher.report(LoopError::Scheduler("distinct control failure".to_owned()));
    });
    assert_eq!(logs.matches("original failure").count(), 1, "{logs}");
    assert_eq!(
        logs.matches("distinct control failure").count(),
        1,
        "{logs}"
    );
}

#[test]
fn notification_is_sent_once_across_concurrent_and_later_reports() {
    const TEST: &str = "tests::notification_is_sent_once_across_concurrent_and_later_reports";
    if std::env::var(CHILD).as_deref() == Ok(TEST) {
        let launcher = LauncherFailure::for_parent();
        let first = thread::scope(|scope| {
            let ready = Arc::new(std::sync::Barrier::new(8));
            let reports: Vec<_> = (0..8)
                .map(|_| {
                    let ready = ready.clone();
                    let launcher = &launcher;
                    scope.spawn(move || {
                        ready.wait();
                        launcher.notify()
                    })
                })
                .collect();
            reports
                .into_iter()
                .map(|report| report.join().unwrap())
                .filter(|first| *first)
                .count()
        });
        assert_eq!(first, 1);
        // Start later reports only after the parent consumed the first signal,
        // so standard-signal coalescing cannot hide another notification.
        io::stdin().read_exact(&mut [0_u8; 1]).unwrap();
        thread::scope(|scope| {
            for _ in 0..8 {
                let launcher = &launcher;
                scope.spawn(move || launcher.report("later failure"));
            }
        });
        return;
    }

    let _signal_test = SIGNAL_TEST
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut failure = {
        let _entered = runtime.enter();
        signal(SignalKind::user_defined1()).unwrap()
    };
    let mut rank = test_child(TEST);
    assert!(runtime.block_on(async {
        matches!(
            tokio::time::timeout(SIGNAL_TIMEOUT, failure.recv()).await,
            Ok(Some(()))
        )
    }));
    rank.0.stdin.as_mut().unwrap().write_all(&[1]).unwrap();
    assert!(rank.wait().success());
    assert!(
        runtime.block_on(async {
            tokio::time::timeout(Duration::from_millis(100), failure.recv())
                .await
                .is_err()
        }),
        "the rank sent another notification after the first was consumed"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn notification_regressions_work_with_a_pid_one_launcher() {
    // User namespaces may be disabled by the host or container policy. The
    // ordinary subprocess tests still run there; report why this variant skips.
    let probe = Command::new("unshare")
        .args([
            "--user",
            "--map-root-user",
            "--pid",
            "--fork",
            "--kill-child=SIGKILL",
            "true",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    let probe = match probe {
        Ok(probe) => probe,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            eprintln!("skipping PID 1 regression: unshare is unavailable");
            return;
        }
        Err(error) => panic!("could not start PID namespace probe: {error}"),
    };
    if !TestChild(probe).wait().success() {
        eprintln!("skipping PID 1 regression: user/PID namespaces are unavailable");
        return;
    }
    for test in [
        "tests::serving_error_notifies_launcher_before_teardown",
        "data_plane::tests::a_failed_completion_notifies_the_launcher_before_logging",
    ] {
        let child = Command::new("unshare")
            .args([
                "--user",
                "--map-root-user",
                "--pid",
                "--fork",
                "--kill-child=SIGKILL",
            ])
            .arg(std::env::current_exe().unwrap())
            .args(["--exact", test])
            .env_remove(CHILD)
            .spawn()
            .unwrap();
        assert!(
            TestChild(child).wait().success(),
            "PID 1 regression failed: {test}"
        );
    }
}

#[test]
fn python_trace_provider_loads_and_falls_back_on_error() {
    use pyo3::{prelude::*, types::PyModule};
    use std::ffi::CString;
    Python::initialize();
    Python::attach(|py| {
        let code = CString::new(
            r#"
def resolve(rid):
    if rid == 'error': raise ValueError('bad context')
    if rid == 'none': return None
    if rid == 'wrong': return 123
    if rid == 'root': return ('1' * 32, None)
    return ('1' * 32, '2' * 16)
trace_provider = resolve
"#,
        )
        .unwrap();
        let module =
            PyModule::from_code(py, &code, c"ts_context_test.py", c"ts_context_test").unwrap();
        let modules = py.import("sys").unwrap().getattr("modules").unwrap();
        modules.set_item("ts_context_test", &module).unwrap();
        let provider = super::trace_provider(module.as_any()).unwrap().unwrap();
        assert_eq!(
            provider("valid"),
            Some(("1".repeat(32), Some("2".repeat(16))))
        );
        assert_eq!(provider("root"), Some(("1".repeat(32), None)));
        assert_eq!(provider("none"), None);
        assert_eq!(provider("error"), None);
        assert_eq!(provider("wrong"), None);
        module.setattr("trace_provider", 123).unwrap();
        assert!(super::trace_provider(module.as_any()).is_err());
        module.setattr("trace_provider", py.None()).unwrap();
        assert!(super::trace_provider(module.as_any()).unwrap().is_none());
        modules.del_item("ts_context_test").unwrap();
    });
}

#[test]
fn logging_mode_is_explicit_and_validated() {
    let args = |mode: &str| {
        [
            "--python",
            "python3",
            "--control-fd",
            "3",
            "--enable-async-logging",
            mode,
        ]
        .map(str::to_owned)
        .into_iter()
    };
    assert!(parse_args(args("true")).unwrap().enable_async_logging);
    assert!(!parse_args(args("false")).unwrap().enable_async_logging);
    assert!(parse_args(args("yes")).is_err());
    assert!(parse_args(args("true").take(4)).is_err());
}
