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

//! Log I/O stays off serving threads when async logging is selected.

use std::io::Write;

use pyo3::{prelude::*, types::PyDict};
use tracing_appender::non_blocking::{NonBlockingBuilder, WorkerGuard};
use tracing_subscriber::{fmt::writer::BoxMakeWriter, EnvFilter};

const QUEUE_CAPACITY: usize = 8192;

fn writer<W: Write + Send + 'static>(
    make_writer: impl Fn() -> W + Send + Sync + 'static,
    enable_async_logging: bool,
) -> (BoxMakeWriter, Option<WorkerGuard>) {
    if enable_async_logging {
        let (writer, guard) = NonBlockingBuilder::default()
            .buffered_lines_limit(QUEUE_CAPACITY)
            .lossy(true)
            .finish(make_writer());
        (BoxMakeWriter::new(writer), Some(guard))
    } else {
        (BoxMakeWriter::new(make_writer), None)
    }
}

pub fn init(enable_async_logging: bool) -> Option<WorkerGuard> {
    let (writer, guard) = writer(std::io::stderr, enable_async_logging);
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_env("TS_HOST_LOG").unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(writer)
        .init();
    guard
}

/// Drain Python logs before returning from the host, including bootstrap
/// failures and encode ranks. Construct only after the interpreter starts.
pub struct PythonLogGuard;

impl Drop for PythonLogGuard {
    fn drop(&mut self) {
        let stopped = Python::attach(|py| -> PyResult<()> {
            let modules = py.import("sys")?.getattr("modules")?;
            // A failed bootstrap may never have imported TokenSpeed. Do not
            // import it for the first time during error cleanup.
            if let Some(logging) = modules.cast::<PyDict>()?.get_item("tokenspeed._logging")? {
                logging.call_method0("stop_async_logging")?;
            }
            Ok(())
        });
        if let Err(error) = stopped {
            tracing::warn!(%error, "could not drain Python logs");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io,
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            mpsc, Arc, Mutex,
        },
        thread,
        time::{Duration, Instant},
    };

    use super::*;

    #[derive(Clone)]
    struct BlockingWriter {
        entered: mpsc::Sender<()>,
        resume: Arc<Mutex<mpsc::Receiver<()>>>,
        blocked: Arc<AtomicBool>,
        output: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for BlockingWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if !self.blocked.swap(true, Ordering::SeqCst) {
                self.entered.send(()).unwrap();
                self.resume
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(5))
                    .map_err(io::Error::other)?;
            }
            self.output.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn blocked_output_only_stalls_synchronous_producers() {
        for enabled in [false, true] {
            let (entered, ready) = mpsc::channel();
            let (release, resume) = mpsc::channel();
            let (done, finished) = mpsc::channel();
            let output = Arc::new(Mutex::new(Vec::new()));
            let stream = BlockingWriter {
                entered,
                resume: Arc::new(Mutex::new(resume)),
                blocked: Arc::new(AtomicBool::new(false)),
                output: output.clone(),
            };
            let producer = thread::spawn(move || {
                let (writer, guard) = writer(move || stream.clone(), enabled);
                let subscriber = tracing_subscriber::fmt()
                    .without_time()
                    .with_ansi(false)
                    .with_writer(writer)
                    .finish();
                tracing::subscriber::with_default(subscriber, || {
                    for index in 0..20 {
                        tracing::info!(index, "record");
                    }
                });
                done.send(()).unwrap();
                drop(guard);
            });
            ready.recv_timeout(Duration::from_secs(5)).unwrap();
            if enabled {
                finished.recv_timeout(Duration::from_secs(1)).unwrap();
            } else {
                assert!(finished.try_recv().is_err());
            }
            release.send(()).unwrap();
            producer.join().unwrap();
            let output = output.lock().unwrap();
            assert_eq!(String::from_utf8_lossy(&output).lines().count(), 20);
        }
    }

    #[derive(Clone)]
    struct PeriodicWriter {
        writes: Arc<AtomicUsize>,
        output: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for PeriodicWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if (self.writes.fetch_add(1, Ordering::SeqCst) + 1) % 10 == 0 {
                thread::sleep(Duration::from_millis(50));
            }
            self.output.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn periodic_slow_write_measurement() {
        for enabled in [false, true] {
            let output = Arc::new(Mutex::new(Vec::new()));
            let stream = PeriodicWriter {
                writes: Arc::new(AtomicUsize::new(0)),
                output: output.clone(),
            };
            let (writer, guard) = writer(move || stream.clone(), enabled);
            let subscriber = tracing_subscriber::fmt()
                .without_time()
                .with_ansi(false)
                .with_writer(writer)
                .finish();
            let mut max_log_call = Duration::ZERO;
            let started = Instant::now();
            tracing::subscriber::with_default(subscriber, || {
                for index in 0..40 {
                    let before = Instant::now();
                    tracing::info!(index, "record");
                    max_log_call = max_log_call.max(before.elapsed());
                    thread::sleep(Duration::from_millis(1));
                }
            });
            let producer_time = started.elapsed();
            drop(guard);
            assert_eq!(
                String::from_utf8_lossy(&output.lock().unwrap())
                    .lines()
                    .count(),
                40
            );
            eprintln!(
                "Rust async={enabled}: producer={producer_time:?}, max_log_call={max_log_call:?}, \
                 records=40/40; 50ms write stall every 10 records"
            );
        }
    }

    #[test]
    fn python_logs_are_drained_on_success_and_error_without_finalizing() {
        std::env::set_var("PYTHONHOME", env!("TS_HOST_TEST_PYTHONHOME"));
        Python::initialize();
        Python::attach(|py| {
            py.import("sys")
                .unwrap()
                .getattr("path")
                .unwrap()
                .call_method1(
                    "insert",
                    (0, concat!(env!("CARGO_MANIFEST_DIR"), "/../../../python")),
                )
                .unwrap();
        });
        for fail in [false, true] {
            let result: Result<(), ()> = {
                let _guard = PythonLogGuard;
                Python::attach(|py| {
                    py.run(
                        c"
import io, logging, time
from tokenspeed._logging import configure_root_logger
class SlowStream(io.StringIO):
    def write(self, text):
        time.sleep(0.005)
        return super().write(text)
stream = SlowStream()
configure_root_logger(level=logging.INFO, log_format='%(message)s', datefmt='',
                      stream=stream, enable_async_logging=True)
for i in range(20):
    logging.info(f'record {i}')
",
                        None,
                        None,
                    )
                    .unwrap();
                });
                if fail {
                    Err(())
                } else {
                    Ok(())
                }
            };
            assert_eq!(result.is_err(), fail);
            Python::attach(|py| {
                let lines: Vec<String> = py
                    .eval(c"stream.getvalue().splitlines()", None, None)
                    .unwrap()
                    .extract()
                    .unwrap();
                assert_eq!(
                    lines,
                    (0..20).map(|i| format!("record {i}")).collect::<Vec<_>>()
                );
            });
        }
    }
}
