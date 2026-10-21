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

//! The Rust metrics against the Python engine's (`tests/fixtures/`): the
//! same scenario's multiprocess files read back as the same exposition, and
//! the relay's message format.
//!
//! Reading the files back needs a Python with `prometheus_client`: the one
//! `TS_TEST_PYTHON` or `PYO3_PYTHON` names, else `python3`. Without it the
//! exposition test reports a skip.

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use ts_metrics::{
    EngineLabels, EngineMetrics, Exporter, KvTransferMetrics, MetricEvent, MetricsConfig,
};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn python() -> String {
    std::env::var("TS_TEST_PYTHON")
        .or_else(|_| std::env::var("PYO3_PYTHON"))
        .unwrap_or_else(|_| "python3".to_owned())
}

fn scratch_dir(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("ts-metrics-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

/// `generate.py`'s `_run_scenario`, through the Rust metrics.
fn run_scenario(directory: &Path) {
    let exporter = Exporter::new(&MetricsConfig {
        prometheus_dir: Some(directory.to_owned()),
        relay_endpoint: None,
    })
    .unwrap();
    let mut engine = EngineMetrics::new(
        &exporter,
        &EngineLabels {
            model_name: "m".to_owned(),
            app_key: String::new(),
            dp_rank: 0,
        },
    );
    engine.record_scheduler_iteration(3, 2, 10, 40, 17);
    engine.record_scheduler_iteration(1, 0, 4, 40, 0);
    engine.record_spec_decode_step(4, 5, 3);
    engine.record_jit_compiles(2, 0.5);
    engine.record_nan_abort();
    engine.record_capacity_abort();
    engine.record_preemptions(2);
    engine.observe_request_queue_time(0.4);
    engine.observe_request_queue_time(9000.0);

    let transfer = KvTransferMetrics::new(&exporter, "m", "");
    transfer.observe_latency(0.003);
    transfer.record_timeout("bootstrapping");
    transfer.record_timeout("waiting_for_input");
    transfer.record_failure();
}

#[test]
fn the_multiprocess_files_read_back_as_the_python_exposition() {
    let directory = scratch_dir("exposition");
    run_scenario(&directory);
    let rendered = Command::new(python())
        .arg(fixtures().join("generate.py"))
        .arg("--render")
        .arg(&directory)
        .output()
        .expect("running Python");
    if !rendered.status.success() {
        let stderr = String::from_utf8_lossy(&rendered.stderr);
        if stderr.contains("No module named 'prometheus_client'") {
            eprintln!("skipped: {} has no prometheus_client", python());
            return;
        }
        panic!("rendering failed:\n{stderr}");
    }
    let expected = std::fs::read_to_string(fixtures().join("exposition.txt")).unwrap();
    let rendered = String::from_utf8(rendered.stdout).unwrap();
    let missing: Vec<&str> = expected
        .lines()
        .filter(|line| !rendered.contains(line))
        .collect();
    let extra: Vec<&str> = rendered
        .lines()
        .filter(|line| !expected.contains(line))
        .collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "missing: {missing:#?}\nextra: {extra:#?}"
    );
    assert_eq!(rendered, expected);
    std::fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn a_relay_message_is_the_python_event_array() {
    let event = MetricEvent {
        timestamp_ns: 1_700_000_000_123_456_789,
        metric: "tokenspeed:kv_transfer_timeouts_total".to_owned(),
        operation: "inc",
        value: 1.0,
        labels: vec![
            ("model_name".to_owned(), "m".to_owned()),
            ("app_key".to_owned(), String::new()),
            ("stage".to_owned(), "bootstrapping".to_owned()),
        ],
        weight: 1,
    };
    let expected = std::fs::read_to_string(fixtures().join("relay_event.hex")).unwrap();
    let encoded: String = event
        .encode()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(encoded, expected.trim());
}
