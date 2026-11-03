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
    EngineLabels, EngineMetrics, Exporter, KvTransferMetrics, L3ExecutorSnapshot, L3Labels,
    L3Metrics, L3OperationSnapshot, L3SchedulerStats, MetricEvent, MetricsConfig,
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

/// The sorted exposition of the multiprocess files in `directory`, or None
/// without `prometheus_client`.
fn render(directory: &Path) -> Option<String> {
    let rendered = Command::new(python())
        .arg(fixtures().join("generate.py"))
        .arg("--render")
        .arg(directory)
        .output()
        .expect("running Python");
    if !rendered.status.success() {
        let stderr = String::from_utf8_lossy(&rendered.stderr);
        if stderr.contains("No module named 'prometheus_client'") {
            eprintln!("skipped: {} has no prometheus_client", python());
            return None;
        }
        panic!("rendering failed:\n{stderr}");
    }
    Some(String::from_utf8(rendered.stdout).unwrap())
}

#[test]
fn l3_snapshots_export_their_growth_once_and_keep_quarantine_visible() {
    let directory = scratch_dir("l3");
    let exporter = Exporter::new(&MetricsConfig {
        prometheus_dir: Some(directory.clone()),
        relay_endpoint: None,
    })
    .unwrap();
    let mut metrics = L3Metrics::new(
        &exporter,
        &L3Labels {
            model_name: "m".to_owned(),
            app_key: String::new(),
            dp_rank: 0,
            tp_rank: 1,
            engine_role: "null".to_owned(),
        },
    );
    let counts = |pairs: &[(&str, f64)]| -> Vec<(String, f64)> {
        pairs.iter().map(|&(k, v)| (k.to_owned(), v)).collect()
    };
    let io = L3ExecutorSnapshot {
        operations: vec![L3OperationSnapshot {
            kind: "get".to_owned(),
            events: counts(&[
                ("accepted", 2.0),
                ("backend_calls", 1.0),
                ("completed", 2.0),
                ("cancelled_queued", 1.0),
                ("unretired", 1.0),
            ]),
            failed_keys: 2.0,
            operations: counts(&[("queued", 0.0), ("running", 1.0)]),
            bytes: counts(&[("queued", 0.0), ("running", 4096.0)]),
            ..L3OperationSnapshot::default()
        }],
        health_state: "open".to_owned(),
        health_epoch: 3.0,
        unsafe_io: true,
    };
    let scheduler = L3SchedulerStats {
        operations: 1.0,
        held_blocks: 2.0,
        quarantined: 1.0,
        negative_keys: 1.0,
        wait_timeouts: 1.0,
        pressure_gate_closed: true,
        pressure_bypasses: 3.0,
        pressure_dropped_backup_keys: 7.0,
        overflow_dropped_backup_keys: 2.0,
        unavailable_dropped_backup_keys: 3.0,
        disabled_dropped_backup_keys: 4.0,
        cleanup_dropped_backup_keys: 5.0,
        ..L3SchedulerStats::default()
    };
    // Snapshots are cumulative: exporting one twice adds nothing.
    metrics.record(&io, &scheduler);
    metrics.record(&io, &scheduler);
    drop(metrics);
    let Some(rendered) = render(&directory) else {
        return;
    };
    let rank = [
        ("app_key", ""),
        ("dp_rank", "0"),
        ("engine_role", "null"),
        ("model_name", "m"),
        ("tp_rank", "1"),
    ];
    for (metric, extra, value) in [
        (
            "tokenspeed:l3_io_events_total",
            &[("kind", "get"), ("event", "completed")][..],
            2.0,
        ),
        (
            "tokenspeed:l3_io_events_total",
            &[("kind", "get"), ("event", "unretired")],
            1.0,
        ),
        (
            "tokenspeed:l3_io_keys_total",
            &[("kind", "get"), ("result", "failure")],
            2.0,
        ),
        (
            "tokenspeed:l3_io_buffer_bytes",
            &[("kind", "get"), ("state", "running")],
            4096.0,
        ),
        ("tokenspeed:l3_backend_state", &[], 1.0),
        ("tokenspeed:l3_backend_epoch", &[], 3.0),
        ("tokenspeed:l3_io_quarantined", &[], 1.0),
        ("tokenspeed:l3_pressure_bypasses_total", &[], 3.0),
        ("tokenspeed:l3_pressure_dropped_backup_keys_total", &[], 7.0),
        (
            "tokenspeed:l3_preparation_expirations_total",
            &[("phase", "waiting")],
            1.0,
        ),
        (
            "tokenspeed:l3_dropped_backup_keys_total",
            &[("reason", "pressure")],
            7.0,
        ),
        (
            "tokenspeed:l3_dropped_backup_keys_total",
            &[("reason", "overflow")],
            2.0,
        ),
        (
            "tokenspeed:l3_dropped_backup_keys_total",
            &[("reason", "host_unavailable")],
            3.0,
        ),
        (
            "tokenspeed:l3_dropped_backup_keys_total",
            &[("reason", "write_disabled")],
            4.0,
        ),
        (
            "tokenspeed:l3_dropped_backup_keys_total",
            &[("reason", "cleanup")],
            5.0,
        ),
        (
            "tokenspeed:l3_scheduler_resources",
            &[("resource", "held_blocks")],
            2.0,
        ),
        (
            "tokenspeed:l3_scheduler_resources",
            &[("resource", "pressure_gate_closed")],
            1.0,
        ),
    ] {
        let mut labels: Vec<(&str, &str)> = rank.iter().chain(extra).copied().collect();
        labels.sort_unstable();
        let rendered_labels: Vec<String> = labels
            .iter()
            .map(|(name, value)| format!("{name}=\"{value}\""))
            .collect();
        let line = format!("{metric}{{{}}} {value:?}", rendered_labels.join(","));
        assert!(rendered.contains(&line), "no {line} in:\n{rendered}");
    }
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
