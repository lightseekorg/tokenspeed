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

//! The executor against a real file with direct I/O, through both backends.
//! The file lives under `TS_DISK_TEST_DIR`, or the temporary directory,
//! which must be on a file system that supports `O_DIRECT` (not tmpfs).

use std::{
    path::PathBuf,
    sync::mpsc,
    time::{Duration, Instant},
};

use ts_disk::{
    Backend, Direction, DiskFile, Executor, ExecutorConfig, Io, IoError, Job, Memory, SubmitError,
    ALIGNMENT,
};

const FILE_BYTES: u64 = 1 << 20;

/// A file removed when the test ends.
struct ScratchFile(PathBuf);

impl ScratchFile {
    fn new(name: &str) -> Self {
        let dir = std::env::var_os("TS_DISK_TEST_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        Self(dir.join(format!("ts-disk-{}-{name}", std::process::id())))
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn config(backend: Backend) -> ExecutorConfig {
    ExecutorConfig {
        backend,
        queue_depth: 8,
        write_queue_depth: 4,
        max_io_bytes: 4 * ALIGNMENT,
        write_bytes_per_second: 1 << 30,
    }
}

fn start(scratch: &ScratchFile, memory: &Memory, config: ExecutorConfig) -> Executor {
    let file = DiskFile::create(&scratch.0, FILE_BYTES).expect("a direct-I/O file");
    Executor::start(file, &[(memory.address(), memory.size())], config).expect("an executor")
}

/// Run `job` and wait for its outcome.
fn run(executor: &Executor, job: Job) -> Result<(), IoError> {
    let (tx, rx) = mpsc::channel();
    executor
        .submit(job, move |outcome| tx.send(outcome).unwrap())
        .expect("a valid job");
    rx.recv_timeout(Duration::from_secs(30))
        .expect("the job reports")
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn round_trip(backend: Backend, name: &str) {
    let scratch = ScratchFile::new(name);
    let mut memory = Memory::map(16 * ALIGNMENT).unwrap();
    let executor = start(&scratch, &memory, config(backend));
    // Two ranges, one longer than the largest request.
    let first = pattern(6 * ALIGNMENT, 1);
    let second = pattern(2 * ALIGNMENT, 7);
    memory.write_at(0, &first);
    memory.write_at(8 * ALIGNMENT, &second);
    let base = memory.address();
    let ios = vec![
        Io {
            file_offset: 3 * ALIGNMENT as u64,
            address: base,
            len: first.len(),
        },
        Io {
            file_offset: 100 * ALIGNMENT as u64,
            address: base + 8 * ALIGNMENT,
            len: second.len(),
        },
    ];
    run(
        &executor,
        Job {
            direction: Direction::Write,
            ios: ios.clone(),
        },
    )
    .unwrap();
    memory.write_at(0, &vec![0; memory.size()]);
    run(
        &executor,
        Job {
            direction: Direction::Read,
            ios,
        },
    )
    .unwrap();
    let mut read = vec![0; first.len()];
    memory.read_at(0, &mut read);
    assert_eq!(read, first);
    let mut read = vec![0; second.len()];
    memory.read_at(8 * ALIGNMENT, &mut read);
    assert_eq!(read, second);
    let stats = executor.stats();
    assert_eq!((stats.read_jobs, stats.write_jobs), (1, 1));
    assert_eq!(stats.read_bytes, (first.len() + second.len()) as u64);
}

#[test]
fn bytes_round_trip_through_io_uring() {
    round_trip(Backend::IoUring, "uring-round-trip");
}

#[test]
fn bytes_round_trip_through_threads() {
    round_trip(Backend::Psync, "psync-round-trip");
}

#[test]
fn ranges_that_are_unaligned_unregistered_or_past_the_file_are_refused() {
    let scratch = ScratchFile::new("refused");
    let memory = Memory::map(4 * ALIGNMENT).unwrap();
    let executor = start(&scratch, &memory, config(Backend::Psync));
    let base = memory.address();
    let submit = |io: Io| {
        executor.submit(
            Job {
                direction: Direction::Read,
                ios: vec![io],
            },
            |_| {},
        )
    };
    let unaligned = Io {
        file_offset: 512,
        address: base,
        len: ALIGNMENT,
    };
    assert_eq!(submit(unaligned), Err(SubmitError::Misaligned(unaligned)));
    let past = Io {
        file_offset: FILE_BYTES,
        address: base,
        len: ALIGNMENT,
    };
    assert_eq!(submit(past), Err(SubmitError::PastFile(past)));
    let outside = Io {
        file_offset: 0,
        address: base + 2 * ALIGNMENT,
        len: 4 * ALIGNMENT,
    };
    assert_eq!(submit(outside), Err(SubmitError::Unregistered(outside)));
    assert_eq!(
        executor.submit(
            Job {
                direction: Direction::Read,
                ios: Vec::new(),
            },
            |_| {},
        ),
        Err(SubmitError::Empty)
    );
}

#[test]
fn writes_reach_the_file_no_faster_than_the_budget() {
    let scratch = ScratchFile::new("paced");
    let memory = Memory::map(80 * ALIGNMENT).unwrap();
    // 1 MiB/s with 64 KiB requests: the burst is one request, so the other
    // 256 KiB take at least a quarter of a second.
    let executor = start(
        &scratch,
        &memory,
        ExecutorConfig {
            max_io_bytes: 16 * ALIGNMENT,
            write_bytes_per_second: 1 << 20,
            ..config(Backend::IoUring)
        },
    );
    let started = Instant::now();
    run(
        &executor,
        Job {
            direction: Direction::Write,
            ios: vec![Io {
                file_offset: 0,
                address: memory.address(),
                len: 80 * ALIGNMENT,
            }],
        },
    )
    .unwrap();
    assert!(started.elapsed() >= Duration::from_millis(240));
}

#[test]
fn a_cancelled_write_issues_nothing_more() {
    let scratch = ScratchFile::new("cancelled");
    let mut memory = Memory::map(3 * ALIGNMENT).unwrap();
    memory.write_at(0, &pattern(3 * ALIGNMENT, 3));
    // One page per second: after the first, each request waits a second.
    let executor = start(
        &scratch,
        &memory,
        ExecutorConfig {
            max_io_bytes: ALIGNMENT,
            write_bytes_per_second: ALIGNMENT as u64,
            ..config(Backend::IoUring)
        },
    );
    let (tx, rx) = mpsc::channel();
    let job = executor
        .submit(
            Job {
                direction: Direction::Write,
                ios: vec![Io {
                    file_offset: 0,
                    address: memory.address(),
                    len: 3 * ALIGNMENT,
                }],
            },
            move |outcome| tx.send(outcome).unwrap(),
        )
        .unwrap();
    executor.cancel(job);
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(30)).unwrap(),
        Err(IoError::Cancelled)
    );
    // The last two pages never reached the file: reading them back gives
    // the zeros it was allocated with.
    memory.write_at(0, &vec![0xff; 2 * ALIGNMENT]);
    run(
        &executor,
        Job {
            direction: Direction::Read,
            ios: vec![Io {
                file_offset: ALIGNMENT as u64,
                address: memory.address(),
                len: 2 * ALIGNMENT,
            }],
        },
    )
    .unwrap();
    let mut read = vec![0xff; 2 * ALIGNMENT];
    memory.read_at(0, &mut read);
    assert!(read.iter().all(|&byte| byte == 0));
    assert_eq!(executor.stats().cancelled_jobs, 1);
}

#[test]
fn closing_cancels_queued_jobs_and_refuses_new_ones() {
    let scratch = ScratchFile::new("closing");
    let memory = Memory::map(4 * ALIGNMENT).unwrap();
    let executor = start(
        &scratch,
        &memory,
        ExecutorConfig {
            max_io_bytes: ALIGNMENT,
            write_bytes_per_second: ALIGNMENT as u64,
            ..config(Backend::Psync)
        },
    );
    let (tx, rx) = mpsc::channel();
    executor
        .submit(
            Job {
                direction: Direction::Write,
                ios: vec![Io {
                    file_offset: 0,
                    address: memory.address(),
                    len: 4 * ALIGNMENT,
                }],
            },
            move |outcome| tx.send(outcome).unwrap(),
        )
        .unwrap();
    executor.close(Duration::from_secs(30)).unwrap();
    assert_eq!(rx.try_recv().unwrap(), Err(IoError::Cancelled));
    assert_eq!(
        executor.submit(
            Job {
                direction: Direction::Read,
                ios: vec![Io {
                    file_offset: 0,
                    address: memory.address(),
                    len: ALIGNMENT,
                }],
            },
            |_| {},
        ),
        Err(SubmitError::Closed)
    );
}
