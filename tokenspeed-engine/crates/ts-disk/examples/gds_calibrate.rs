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

//! Measure one logical Disk block per job through the real executor.
//!
//! Unlike disk_bench, this emits JSON, accepts arbitrary aligned block sizes,
//! and never opens an existing file. See test/cache_gds_calibration/README.md.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::mpsc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use ts_disk::{
    ALIGNMENT, Backend, Direction, DiskFile, Executor, ExecutorConfig, Io, Job, JobReport, Memory,
};

struct Scratch(PathBuf);

impl Scratch {
    fn new(parent: &Path) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = parent.join(format!("ts-gds-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).expect("create a new scratch directory");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.0.join("blocks.bin"));
        let _ = fs::remove_dir(&self.0);
    }
}

fn run(args: &[String]) -> (usize, f64) {
    assert_eq!(
        args.len(),
        10,
        "usage: gds_calibrate <scratch-dir> <file-bytes> <block-bytes> \
         <inflight> <queue-depth> <max-io-bytes> <uring|psync> <warmup-seconds> <seconds>"
    );
    let file_budget: u64 = args[2].parse().expect("file bytes");
    let block: usize = args[3].parse().expect("block bytes");
    let inflight: usize = args[4].parse().expect("inflight jobs");
    let queue_depth: usize = args[5].parse().expect("queue depth");
    let max_io_bytes: usize = args[6].parse().expect("maximum I/O bytes");
    let backend = match args[7].as_str() {
        "uring" => Backend::IoUring,
        "psync" => Backend::Psync,
        _ => panic!("backend must be uring or psync"),
    };
    let warmup: f64 = args[8].parse().expect("warmup seconds");
    let seconds: f64 = args[9].parse().expect("measurement seconds");
    assert!(block > 0 && block % ALIGNMENT == 0, "block must be aligned");
    assert!(inflight > 0 && queue_depth > 0);
    assert!(max_io_bytes > 0 && max_io_bytes % ALIGNMENT == 0);
    assert!(warmup.is_finite() && warmup > 0.0 && seconds.is_finite() && seconds > 0.0);
    let blocks = file_budget / block as u64;
    assert!(
        blocks >= inflight as u64,
        "file must fit every in-flight block"
    );
    let file_bytes = blocks * block as u64;
    let scratch = Scratch::new(Path::new(&args[1]));
    let mut memory = Memory::map(inflight.checked_mul(block).expect("buffer size")).unwrap();
    // Incompressible staging data avoids benchmarking only zero-filled extents.
    let mut random_state = 0x9e37_79b9_7f4a_7c15u64;
    let mut random = || {
        random_state ^= random_state << 13;
        random_state ^= random_state >> 7;
        random_state ^= random_state << 17;
        random_state
    };
    let mut page = [0u8; ALIGNMENT];
    for offset in (0..memory.size()).step_by(ALIGNMENT) {
        for chunk in page.chunks_exact_mut(8) {
            chunk.copy_from_slice(&random().to_le_bytes());
        }
        memory.write_at(offset, &page);
    }
    let file = DiskFile::create(&scratch.0.join("blocks.bin"), file_bytes).unwrap();
    let executor = Executor::start(
        file,
        &[(memory.address(), memory.size())],
        ExecutorConfig {
            backend,
            queue_depth,
            write_queue_depth: queue_depth,
            max_io_bytes,
            write_bytes_per_second: u64::MAX / 2,
        },
    )
    .expect("start Disk executor; no buffered-I/O fallback");
    let (tx, rx) = mpsc::channel::<(usize, Instant, Instant, bool)>();
    let submit = |slot: usize, direction: Direction, index: u64| {
        let tx = tx.clone();
        let started = Instant::now();
        executor
            .submit(
                Job {
                    direction,
                    ios: vec![Io {
                        file_offset: index * block as u64,
                        address: memory.address() + slot * block,
                        len: block,
                    }],
                },
                move |report: JobReport| {
                    let _ = tx.send((slot, started, Instant::now(), report.outcome.is_ok()));
                },
            )
            .expect("submit a valid job");
    };
    // Initialize every byte through direct writes before collecting reads.
    let mut next = 0;
    for slot in 0..inflight {
        submit(slot, Direction::Write, next);
        next += 1;
    }
    for _ in 0..blocks {
        let (slot, _, _, ok) = rx.recv().expect("write completion");
        assert!(ok, "write failed");
        if next < blocks {
            submit(slot, Direction::Write, next);
            next += 1;
        }
    }
    let mut measure = |duration: f64| {
        let started = Instant::now();
        let deadline = started + Duration::from_secs_f64(duration);
        for slot in 0..inflight {
            submit(slot, Direction::Read, random() % blocks);
        }
        let mut active = inflight;
        let mut latencies = Vec::new();
        while active > 0 {
            let (slot, issued, finished, ok) = rx.recv().expect("read completion");
            assert!(ok, "read failed");
            latencies.push(finished.duration_since(issued).as_secs_f64());
            if Instant::now() < deadline {
                submit(slot, Direction::Read, random() % blocks);
            } else {
                active -= 1;
            }
        }
        (started.elapsed().as_secs_f64(), latencies)
    };
    let _ = measure(warmup);
    let (elapsed, mut latencies) = measure(seconds);
    latencies.sort_unstable_by(f64::total_cmp);
    let count = latencies.len();
    let mean = latencies.iter().sum::<f64>() / count as f64;
    let p50 = latencies[(count - 1) / 2];
    let p95 = latencies[(count - 1) * 95 / 100];
    println!(
        "{{\"schema\":1,\"backend\":\"{}\",\"direct_io\":true,\
         \"file_bytes\":{file_bytes},\"block_bytes\":{block},\
         \"inflight\":{inflight},\"queue_depth\":{queue_depth},\
         \"max_io_bytes\":{max_io_bytes},\"completed_blocks\":{count},\
         \"elapsed_s\":{elapsed},\"mean_latency_s\":{mean},\
         \"p50_latency_s\":{p50},\"p95_latency_s\":{p95}}}",
        args[7]
    );
    (count, elapsed)
}

fn main() {
    run(&std::env::args().collect::<Vec<_>>());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_split_blocks_and_only_removes_its_own_files() {
        let parent = Scratch::new(&std::env::temp_dir());
        let sentinel = parent.0.join("existing-cache.bin");
        fs::write(&sentinel, b"unchanged").unwrap();
        let args: Vec<String> = [
            "gds_calibrate",
            parent.0.to_str().unwrap(),
            "70000",
            "12288",
            "2",
            "2",
            "4096",
            "psync",
            "0.01",
            "0.02",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        let (count, elapsed) = run(&args);
        assert!(count > 0 && elapsed >= 0.02);
        assert_eq!(fs::read(&sentinel).unwrap(), b"unchanged");
        assert_eq!(fs::read_dir(&parent.0).unwrap().count(), 1);
        fs::remove_file(sentinel).unwrap();
    }

    #[test]
    #[should_panic(expected = "block must be aligned")]
    fn refuses_unaligned_geometry() {
        let args: Vec<String> = [
            "gds_calibrate",
            "/unused",
            "65536",
            "4097",
            "1",
            "1",
            "4096",
            "psync",
            "0.01",
            "0.02",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        run(&args);
    }
}
