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

//! CPU-only measurements of the real Rust ingress, output and IPC paths.
mod support;
use support::{Config, Logprobs};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(args.len(),13,"SUITE TRANSPORT BATCH PROMPT STEPS LOGPROBS TOP_K PERIOD_US CONSUMER_DELAY_US CLIENT_THREADS WARMUPS REPEATS PHASE_TIMERS");
    let number = |i: usize| args[i].parse::<usize>().expect("nonnegative integer");
    let config = Config {
        suite: args[0].clone(),
        transport: args[1].clone(),
        batch: number(2),
        prompt: number(3),
        steps: number(4),
        logprobs: match args[5].as_str() {
            "none" => Logprobs::None,
            "output" => Logprobs::Output,
            "prompt" => Logprobs::Prompt,
            _ => panic!("LOGPROBS: none/output/prompt"),
        },
        top_k: number(6),
        period_us: number(7) as u64,
        consumer_delay_us: number(8) as u64,
        client_threads: number(9),
        warmups: number(10),
        repeats: number(11),
        phase_timers: match args[12].as_str() {
            "on" => true,
            "off" => false,
            _ => panic!("PHASE_TIMERS: on/off"),
        },
    };
    config.validate().unwrap();
    let mut samples = Vec::with_capacity(config.repeats);
    for i in 0..config.warmups + config.repeats {
        eprintln!("sample {}/{}", i + 1, config.warmups + config.repeats);
        let sample = match config.suite.as_str() {
            "stages" => support::run_stages(&config),
            "pipeline" => support::pipeline::run(&config),
            _ => unreachable!(),
        };
        if i >= config.warmups {
            samples.push(sample);
        }
    }
    println!(
        "{}",
        serde_json::json!({"benchmark":"rust_engine_cpu","config":config.json(),"samples":samples})
    );
}
