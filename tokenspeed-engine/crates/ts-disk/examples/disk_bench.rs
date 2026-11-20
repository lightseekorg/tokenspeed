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

//! Cold-read baseline of a Disk tier file: fill it with sequential writes,
//! then read random blocks with a fixed number of jobs in flight and report
//! bandwidth and job latency. Direct I/O reads the device, not the page
//! cache.
//!
//! `cargo run --release -p tokenspeed-disk --example disk_bench -- \
//!    <file> <file GiB> <block KiB> <jobs in flight> <uring|psync> <seconds>`

use std::{
    path::PathBuf,
    sync::mpsc,
    time::{Duration, Instant},
};

use ts_disk::{Backend, Direction, DiskFile, Executor, ExecutorConfig, Io, Job, Memory, ALIGNMENT};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        eprintln!(
            "usage: {} <file> <file GiB> <block KiB> <jobs in flight> <uring|psync> <seconds>",
            args[0]
        );
        std::process::exit(2);
    }
    let path = PathBuf::from(&args[1]);
    let file_bytes = args[2].parse::<u64>().expect("file GiB") << 30;
    let block = args[3].parse::<usize>().expect("block KiB") << 10;
    let jobs = args[4].parse::<usize>().expect("jobs in flight");
    let backend = match args[5].as_str() {
        "uring" => Backend::IoUring,
        "psync" => Backend::Psync,
        other => panic!("unknown backend {other}"),
    };
    let seconds = args[6].parse::<f64>().expect("seconds");
    assert!(block % ALIGNMENT == 0 && file_bytes % block as u64 == 0);

    let memory = Memory::map(jobs * block).expect("staging memory");
    let file = DiskFile::create(&path, file_bytes).expect("the file");
    let executor = Executor::start(
        file,
        &[(memory.address(), memory.size())],
        ExecutorConfig {
            backend,
            queue_depth: jobs.max(32),
            write_queue_depth: jobs.max(32),
            max_io_bytes: block.min(8 << 20),
            write_bytes_per_second: u64::MAX / 2,
        },
    )
    .expect("an executor");

    let blocks = file_bytes / block as u64;
    let (tx, rx) = mpsc::channel::<(usize, Instant, bool)>();
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
                move |outcome| {
                    let _ = tx.send((slot, started, outcome.is_ok()));
                },
            )
            .expect("a valid job");
    };

    // Fill: sequential writes, every block once.
    let started = Instant::now();
    let mut next = 0u64;
    for slot in 0..jobs.min(blocks as usize) {
        submit(slot, Direction::Write, next);
        next += 1;
    }
    let mut done = 0u64;
    while done < blocks {
        let (slot, _, ok) = rx.recv().expect("a completion");
        assert!(ok, "a write failed");
        done += 1;
        if next < blocks {
            submit(slot, Direction::Write, next);
            next += 1;
        }
    }
    let fill = started.elapsed().as_secs_f64();
    println!(
        "fill: {:.2} GiB in {fill:.2} s = {:.2} GB/s",
        file_bytes as f64 / (1u64 << 30) as f64,
        file_bytes as f64 / fill / 1e9
    );

    // Random reads.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut random = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % blocks
    };
    let started = Instant::now();
    let deadline = started + Duration::from_secs_f64(seconds);
    for slot in 0..jobs {
        submit(slot, Direction::Read, random());
    }
    let mut latencies = Vec::new();
    let mut inflight = jobs;
    while inflight > 0 {
        let (slot, issued, ok) = rx.recv().expect("a completion");
        assert!(ok, "a read failed");
        latencies.push(issued.elapsed());
        if Instant::now() < deadline {
            submit(slot, Direction::Read, random());
        } else {
            inflight -= 1;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    latencies.sort_unstable();
    let percentile = |p: f64| latencies[((latencies.len() - 1) as f64 * p) as usize];
    let bytes = latencies.len() as f64 * block as f64;
    println!(
        "read: block {} KiB, {jobs} in flight, {:?}: {:.2} GB/s, {:.0} IOPS, p50 {:?}, p99 {:?}, max {:?}",
        block >> 10,
        backend,
        bytes / elapsed / 1e9,
        latencies.len() as f64 / elapsed,
        percentile(0.5),
        percentile(0.99),
        latencies.last().unwrap(),
    );
    let _ = std::fs::remove_file(&path);
}
