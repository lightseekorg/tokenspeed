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

//! CPU-array diagnostic for the actual Rust result extraction boundary.
//!
//! Only the GPU-dependent plane/event is a stand-in. Its `.tolist()` operations
//! match HostDataPlane.complete and _logprob_lists in rust_host.py. This does
//! not invoke a model, synchronize a GPU event, or replace the Rust extractor.

use super::{complete, interpreter};
use pyo3::{prelude::*, types::PyDict};
use rustix::time::{clock_gettime, ClockId};
use serde_json::{json, Value};
use std::{ffi::CString, path::PathBuf, time::Instant};

const PLANE: &str = r#"
import time
import torch
torch.set_num_threads(1)

class CpuResultPlane:
    def __init__(self, batch, prompt, mode, top_k):
        self.tokens = torch.full((batch,), 120000, dtype=torch.int32, device="cpu")
        self.lengths = torch.ones(batch, dtype=torch.int32, device="cpu")
        self.output_lp = None if mode == 'none' else torch.full((batch,), -0.5, dtype=torch.float32, device="cpu")
        rows = batch * (prompt - 1) if mode == 'prompt' else 0
        self.input_lp = torch.full((rows,), -1.25, dtype=torch.float32, device="cpu") if rows else None
        self.input_top_values = torch.full((rows, top_k), -3.0, dtype=torch.float32, device="cpu") if rows and top_k else None
        self.input_top_ids = torch.full((rows, top_k), 110000, dtype=torch.int64, device="cpu") if rows and top_k else None
        self.output_top_values = torch.full((batch, top_k), -3.0, dtype=torch.float32, device="cpu") if top_k else None
        self.output_top_ids = torch.full((batch, top_k), 110000, dtype=torch.int64, device="cpu") if top_k else None
        self.last_cpu_ns = 0
        self.last_wall_ns = 0

    def logprob_lists(self):
        def top(values, ids):
            return (values.tolist(), ids.tolist()) if values is not None else None
        if self.input_lp is None and self.input_top_values is None and self.output_top_values is None:
            return None
        return (self.input_lp.tolist() if self.input_lp is not None else None,
                top(self.input_top_values, self.input_top_ids),
                top(self.output_top_values, self.output_top_ids))

    def complete(self, launched):
        cpu = time.thread_time_ns()
        wall = time.perf_counter_ns()
        result = (self.tokens.tolist(), self.lengths.tolist(),
                  self.output_lp.tolist() if self.output_lp is not None else None,
                  None, None, None, None, self.logprob_lists(), None)
        self.last_wall_ns = time.perf_counter_ns() - wall
        self.last_cpu_ns = time.thread_time_ns() - cpu
        return result
"#;

fn cpu_ms() -> f64 {
    let t = clock_gettime(ClockId::ThreadCPUTime);
    t.tv_sec as f64 * 1000.0 + t.tv_nsec as f64 / 1e6
}

fn plane<'py>(
    py: Python<'py>,
    batch: usize,
    prompt: usize,
    mode: &str,
    top_k: usize,
) -> Bound<'py, PyAny> {
    let globals = PyDict::new(py);
    py.run(&CString::new(PLANE).unwrap(), Some(&globals), None)
        .unwrap();
    globals
        .get_item("CpuResultPlane")
        .unwrap()
        .unwrap()
        .call1((batch, prompt, mode, top_k))
        .unwrap()
}

fn measured(
    batch: usize,
    prompt: usize,
    mode: &str,
    top_k: usize,
    warmups: usize,
    repeats: usize,
    freeze_gc: bool,
) -> Value {
    assert!(batch > 0 && prompt > 0 && repeats > 0);
    assert!(matches!(mode, "none" | "output" | "prompt"));
    assert!(top_k <= 20 && (top_k == 0 || mode != "none"));
    interpreter();
    Python::attach(|py| {
        let plane = plane(py, batch, prompt, mode, top_k);
        // The opt-in benchmark runs alone in its process. Mirror the serving
        // startup policy; ordinary parallel unit tests do not change global GC.
        if freeze_gc {
            py.run(c"import gc; gc.collect(); gc.freeze(); a,b,_=gc.get_threshold(); gc.set_threshold(a,b,10000)",None,None).unwrap();
        }
        let launched = py.None();
        let mut samples = Vec::with_capacity(repeats);
        for i in 0..warmups + repeats {
            let cpu = cpu_ms();
            let wall = Instant::now();
            let result = complete(&plane, &launched).unwrap();
            let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
            let cpu_ms = cpu_ms() - cpu;
            let python_cpu_ms = plane
                .getattr("last_cpu_ns")
                .unwrap()
                .extract::<u64>()
                .unwrap() as f64
                / 1e6;
            let python_wall_ms = plane
                .getattr("last_wall_ns")
                .unwrap()
                .extract::<u64>()
                .unwrap() as f64
                / 1e6;
            assert!(python_cpu_ms <= cpu_ms + 0.01);
            assert_eq!(result.output_tokens.len(), batch);
            assert_eq!(result.output_tokens[0], 120000);
            assert_eq!(result.output_lengths, vec![1; batch]);
            if mode == "none" {
                assert!(result.output_logprobs.is_none());
            } else {
                assert_eq!(result.output_logprobs.as_ref().unwrap(), &vec![-0.5; batch]);
            }
            let prompt_rows = if mode == "prompt" {
                batch * (prompt - 1)
            } else {
                0
            };
            if prompt_rows > 0 {
                let values = result.input_token_logprobs.as_ref().unwrap();
                assert_eq!(values.len(), prompt_rows);
                assert_eq!(values[0], -1.25);
                assert_eq!(*values.last().unwrap(), -1.25);
            } else {
                assert!(result.input_token_logprobs.is_none());
            }
            if top_k > 0 {
                let (values, ids) = result.output_top_logprobs.as_ref().unwrap();
                assert_eq!(values.len(), batch);
                assert_eq!(values[0], vec![-3.0; top_k]);
                assert_eq!(ids[0], vec![110000; top_k]);
                if prompt_rows > 0 {
                    let (values, ids) = result.input_top_logprobs.as_ref().unwrap();
                    assert_eq!(values.len(), prompt_rows);
                    assert_eq!(ids.len(), prompt_rows);
                    assert_eq!(values[0], vec![-3.0; top_k]);
                    assert_eq!(ids[0], vec![110000; top_k]);
                    assert_eq!(values.last().unwrap(), &vec![-3.0; top_k]);
                    assert_eq!(ids.last().unwrap(), &vec![110000; top_k]);
                }
            } else {
                assert!(result.output_top_logprobs.is_none());
                assert!(result.input_top_logprobs.is_none());
            }
            let drop_start = Instant::now();
            drop(result);
            let rust_result_drop_ms = drop_start.elapsed().as_secs_f64() * 1000.0;
            if i >= warmups {
                samples.push(json!({"wall_ms":wall_ms,"thread_cpu_ms":cpu_ms,
                "python_tolist_wall_ms":python_wall_ms,"python_tolist_cpu_ms":python_cpu_ms,
                "extract_and_python_cleanup_cpu_ms":cpu_ms-python_cpu_ms,"rust_result_drop_wall_ms":rust_result_drop_ms}));
            }
        }
        json!({"benchmark":"rust_host_result_bridge_cpu","batch":batch,"prompt":prompt,"mode":mode,
            "top_k":top_k,"warmups":warmups,"repeats":repeats,"tensor_dtype":"float32/int32/int64","torch_threads":1,
            "serving_gc":freeze_gc,"python":Python::version_str().split_whitespace().next().unwrap(),"torch":py.import("torch").unwrap().getattr("__version__").unwrap().extract::<String>().unwrap(),"samples":samples})
    })
}

#[test]
#[ignore = "optional CPU PyTorch diagnostic; run explicitly with the benchmark environment"]
fn cpu_results_cross_the_real_extractor_with_optional_logprob_columns() {
    for mode in ["none", "output", "prompt"] {
        measured(3, 17, mode, 0, 0, 1, false);
    }
    measured(3, 17, "prompt", 5, 0, 1, false);
    measured(3, 17, "output", 5, 0, 1, false);
}

#[test]
#[ignore = "explicit CPU result-conversion workload; no GPU or model weights"]
fn benchmark_python_result_bridge() {
    let number = |key: &str| {
        std::env::var(key)
            .unwrap_or_else(|_| panic!("missing {key}"))
            .parse::<usize>()
            .unwrap()
    };
    let mode = std::env::var("TS_BRIDGE_MODE").expect("TS_BRIDGE_MODE is required");
    let output =
        PathBuf::from(std::env::var("TS_BRIDGE_OUTPUT").expect("TS_BRIDGE_OUTPUT is required"));
    let report = measured(
        number("TS_BRIDGE_BATCH"),
        number("TS_BRIDGE_PROMPT"),
        &mode,
        number("TS_BRIDGE_TOP_K"),
        number("TS_BRIDGE_WARMUPS"),
        number("TS_BRIDGE_REPEATS"),
        true,
    );
    std::fs::write(output, serde_json::to_vec(&report).unwrap()).unwrap();
}
