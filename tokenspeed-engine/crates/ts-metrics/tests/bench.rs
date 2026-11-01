// Microbenchmarks of the metrics hot path, opt-in with
// `cargo test --release -p tokenspeed-metrics --test bench -- --ignored
// --nocapture`. A/B two checkouts on one machine, several runs each.

use std::time::{Duration, Instant};

use ts_metrics::{EngineLabels, EngineMetrics, Exporter, MetricsConfig};

fn engine(directory: &std::path::Path) -> (std::sync::Arc<Exporter>, EngineMetrics) {
    let exporter = Exporter::new(&MetricsConfig {
        prometheus_dir: Some(directory.to_owned()),
        relay_endpoint: None,
    })
    .unwrap();
    let engine = EngineMetrics::new(
        &exporter,
        &EngineLabels {
            model_name: "m".to_owned(),
            app_key: String::new(),
            dp_rank: 0,
        },
    );
    (exporter, engine)
}

/// One serving round's metric updates — the scheduler snapshot, a verify
/// step, the cache counters and one request's queue wait — plus the
/// round's flush.
#[test]
#[ignore = "a sustained load that A/Bs the metrics path"]
fn bench_metric_round() {
    let directory = std::env::temp_dir().join(format!("ts-metrics-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let (exporter, engine) = engine(&directory);
    let run = Duration::from_secs(4);
    let start = Instant::now();
    let mut rounds = 0usize;
    while start.elapsed() < run {
        engine.record_scheduler_iteration(3, 2, 10, 40, 17);
        engine.record_spec_decode_step(4, 5, 3);
        engine.observe_request_queue_time(1.2);
        engine.observe_request_queue_time(0.4);
        exporter.flush();
        rounds += 1;
    }
    println!(
        "metric round: {:.0} ns/round over {rounds} rounds",
        start.elapsed().as_secs_f64() * 1e9 / rounds as f64
    );
    let _ = std::fs::remove_dir_all(&directory);
}

/// A burst of one metric's updates: the buffered cost per update.
#[test]
#[ignore = "a sustained load that A/Bs the metrics path"]
fn bench_counter_update() {
    let directory =
        std::env::temp_dir().join(format!("ts-metrics-bench-inc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let (exporter, engine) = engine(&directory);
    let run = Duration::from_secs(4);
    let start = Instant::now();
    let mut updates = 0usize;
    while start.elapsed() < run {
        for _ in 0..1000 {
            engine.record_preemptions(1);
        }
        updates += 1000;
    }
    exporter.flush();
    println!(
        "counter update: {:.0} ns/update over {updates} updates",
        start.elapsed().as_secs_f64() * 1e9 / updates as f64
    );
    let _ = std::fs::remove_dir_all(&directory);
}
