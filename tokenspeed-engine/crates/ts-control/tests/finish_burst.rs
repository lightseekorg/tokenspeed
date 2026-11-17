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

//! Synchronized completion bursts through the real output processor and scheduler.
//! See benches/README.md for timing boundaries and explicit configuration.
use pyo3::{prelude::*, types::PyDict};
use rustix::time::{clock_gettime, ClockId};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::CString,
    sync::Once,
    time::Instant,
};
use ts_control::{
    event_loop::Load,
    output::{FinishLog, ForwardResult, OutputProcessor, Role},
    request::RequestState,
    transport::output_batch,
};
use ts_scheduler::{profile, ForwardBatchView, RequestSpec, Scheduler};
use ts_wire::EngineOutput;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Clone, Copy)]
struct Shape {
    batch: usize,
    prompt: usize,
    steps: usize,
    page: usize,
    hit_percent: usize,
    prefix_cache: bool,
    events: bool,
    profile: bool,
}

fn interpreter() {
    static START: Once = Once::new();
    START.call_once(|| {
        std::env::set_var("PYTHONHOME", env!("TS_CONTROL_TEST_PYTHONHOME"));
        Python::initialize();
        Python::attach(|py| {
            let module = ts_scheduler::python_module(py).unwrap();
            py.import("sys")
                .unwrap()
                .getattr("modules")
                .unwrap()
                .set_item("tokenspeed_scheduler.tokenspeed_scheduler_ext", module)
                .unwrap();
            py.import("sys")
                .unwrap()
                .getattr("path")
                .unwrap()
                .call_method1(
                    "insert",
                    (
                        0,
                        concat!(
                            env!("CARGO_MANIFEST_DIR"),
                            "/../../../tokenspeed-scheduler/python"
                        ),
                    ),
                )
                .unwrap();
        });
    });
}

fn scheduler(s: Shape) -> Scheduler {
    interpreter();
    let pages = s.batch * ((s.prompt + s.steps).div_ceil(s.page) + 4) * 2;
    let code = format!(
        r#"
import tokenspeed_scheduler as ts
cfg = ts.SchedulerConfig()
cfg.role = ts.SchedulerConfig.Role.Fused
cfg.prefix_granularity = {page}
cfg.prefix_replay_tokens = 0
cfg.max_scheduled_tokens = {tokens}
cfg.max_batch_size = {batch}
cfg.max_resident_requests = {batch}
cfg.num_device_pages = {pages} + 1
cfg.num_snapshot_pages = 1
cfg.disable_l2_cache = True
cfg.enable_l3_storage = False
cfg.disable_prefix_cache = {disabled}
cfg.enable_kv_cache_events = {events}
cfg.overlap_schedule_depth = 0
cfg.decode_input_tokens = 1
cfg.cache_groups = [ts.CacheGroupConfig(group_id="full_attention", block_granularity={page},
    total_pages={pages} + 1, attn_kind=ts.AttnKind.Full)]
result = ts.Scheduler(cfg)
"#,
        page = s.page,
        tokens = s.batch * s.prompt,
        batch = s.batch,
        disabled = if s.prefix_cache { "False" } else { "True" },
        events = if s.events { "True" } else { "False" }
    );
    Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(&CString::new(code).unwrap(), Some(&globals), None)
            .unwrap();
        Scheduler::from_python(&globals.get_item("result").unwrap().unwrap()).unwrap()
    })
}

fn requests(s: Shape, prompt: usize, steps: usize) -> Vec<RequestSpec> {
    (0..s.batch)
        .map(|row| {
            let mut tokens: Vec<i32> = (0..prompt).map(|pos| (pos % 100000 + 100) as i32).collect();
            // Each row has its own hash chain, including in the cache-seeding run.
            tokens[0] = row as i32 + 110000;
            RequestSpec {
                request_id: format!("{row:08x}-9f09-89be-5c9d-5d65f086d356"),
                session_id: String::new(),
                cache_group_id: String::new(),
                shared_prefix_tokens: Vec::new(),
                tokens,
                max_new_tokens: steps as i32,
                max_cached_prefix_tokens: i32::MAX,
            }
        })
        .collect()
}

fn processor(specs: &[RequestSpec]) -> OutputProcessor {
    let mut p = OutputProcessor::new(
        None,
        1,
        None,
        Role::Plain,
        FinishLog {
            enabled: false,
            request_stats: false,
            ttft_percentage: 0,
        },
        false,
        None,
    );
    for r in specs {
        let mut state = RequestState::new(
            r.tokens.clone(),
            r.max_new_tokens as i64,
            BTreeSet::new(),
            true,
            None,
            false,
        );
        p.assign_admission(&mut state);
        p.register(r.request_id.clone(), state);
    }
    p
}

fn result(batch: usize) -> ForwardResult {
    ForwardResult {
        output_tokens: vec![17; batch],
        output_lengths: vec![1; batch],
        ..ForwardResult::default()
    }
}

fn view(s: &mut Scheduler, batch: usize) -> (ts_scheduler::ExecutionPlan, ForwardBatchView) {
    let plan = s.next_execution_plan(true, false);
    let mut batches = plan.forward_batches();
    assert_eq!(batches.len(), 1);
    let b = batches.pop().unwrap();
    assert_eq!(b.request_ids.len(), batch);
    (plan, b)
}

fn seed(s: &mut Scheduler, shape: Shape) {
    let hit = shape.prompt / shape.page * shape.hit_percent / 100 * shape.page;
    if hit == 0 {
        return;
    }
    let specs = requests(shape, hit, 1);
    let mut p = processor(&specs);
    s.submit_requests(&specs).unwrap();
    let (plan, b) = view(s, shape.batch);
    let committed = p.commit(&b, None, &result(shape.batch)).unwrap();
    assert_eq!(committed.released.len(), shape.batch);
    s.advance(&committed.events).unwrap();
    drop(plan);
    assert!(!s.next_execution_plan(true, false).needs_device());
    s.drain_kv_events();
}

fn cpu_ms() -> f64 {
    let t = clock_gettime(ClockId::ThreadCPUTime);
    t.tv_sec as f64 * 1000.0 + t.tv_nsec as f64 / 1e6
}
fn timed<T>(f: impl FnOnce() -> T) -> (T, Value) {
    let cpu = cpu_ms();
    let start = Instant::now();
    let out = f();
    let wall = start.elapsed().as_secs_f64() * 1000.0;
    let cpu = cpu_ms() - cpu;
    (out, json!({"wall_ms":wall,"thread_cpu_ms":cpu}))
}

fn trial(shape: Shape, check_reuse: bool) -> Value {
    let mut s = scheduler(shape);
    seed(&mut s, shape);
    let specs = requests(shape, shape.prompt, shape.steps);
    let mut p = processor(&specs);
    s.submit_requests(&specs).unwrap();
    drop(specs);
    let model = result(shape.batch);
    let (mut plan, mut batch) = view(&mut s, shape.batch);
    assert_eq!(batch.num_extends, shape.batch);
    let hit = shape.prompt / shape.page * shape.hit_percent / 100 * shape.page;
    assert_eq!(batch.extend_prefix_lens, vec![hit as i32; shape.batch]);
    let mut before_finish_plan = Value::Null;
    for _ in 1..shape.steps {
        let committed = p.commit(&batch, None, &model).unwrap();
        assert!(committed.released.is_empty());
        s.advance(&committed.events).unwrap();
        drop(plan);
        let (next, time) = timed(|| s.next_execution_plan(true, false));
        before_finish_plan = time;
        let mut batches = next.forward_batches();
        assert_eq!(batches.len(), 1);
        batch = batches.pop().unwrap();
        assert_eq!(batch.num_extends, 0);
        assert_eq!(batch.request_ids.len(), shape.batch);
        plan = next;
    }
    // Keep earlier KV events out of the final drain timer.
    s.drain_kv_events();
    let (committed, commit) = timed(|| p.commit(&batch, None, &model).unwrap());
    assert_eq!(committed.released.len(), shape.batch);
    assert_eq!(committed.outputs.len(), shape.batch);
    assert_eq!(p.num_running(), 0);
    for out in &committed.outputs {
        assert_eq!(out.finish_reason, Some("length"));
        assert_eq!(out.completion_tokens, shape.steps as i64);
        assert_eq!(out.output_ids, [17]);
    }
    if shape.profile {
        profile::start();
    }
    let (_, finish) = timed(|| s.advance(&committed.events).unwrap());
    let mut spans: BTreeMap<String, f64> = BTreeMap::new();
    if shape.profile {
        for span in profile::stop().spans {
            *spans.entry(span.name.into_owned()).or_default() +=
                (span.end_ns - span.begin_ns) as f64 / 1e6;
        }
    }
    assert_eq!(s.active_lcm_blocks(), 0);
    let (kv_events, events) = timed(|| s.drain_kv_events());
    let num_events = kv_events.len();
    assert_eq!(num_events > 0, shape.events && shape.prefix_cache);
    // Measure the codec separately here; serving encodes on the publisher thread.
    let (kv_payload, kv_encode) = timed(|| {
        if kv_events.is_empty() {
            Vec::new()
        } else {
            ts_control::kv_events::encode_batch(0.0, &kv_events, 0)
        }
    });
    let kv_event_bytes = kv_payload.len();
    // Tail event publication precedes retirement in the next round's plan.
    let (cleanup_plan, cleanup) = timed(|| s.next_execution_plan(true, false));
    assert!(!cleanup_plan.needs_device());
    assert_eq!(s.sizes().running, 0);
    assert_eq!(s.sizes().waiting, 0);
    let (wire, output_batch_time) = timed(|| output_batch(committed.outputs, 0, Load::default()));
    let (frames, encode) = timed(|| EngineOutput::Batch(Box::new(wire)).to_frames());
    let wire_bytes: usize = frames.iter().map(|frame| frame.len()).sum();
    // These destructors also run in serving, although on different threads.
    let (_, metadata_drop) = timed(|| {
        drop((
            plan,
            cleanup_plan,
            batch,
            model,
            committed.events,
            committed.released,
        ))
    });
    let (_, wire_drop) = timed(|| drop((frames, kv_events, kv_payload)));
    if check_reuse {
        // Reuse the IDs and all full prompt pages before the final replay token.
        let specs = requests(shape, shape.prompt, 1);
        s.submit_requests(&specs).unwrap();
        let (_, b) = view(&mut s, shape.batch);
        let expected = if shape.prefix_cache {
            (shape.prompt - 1) / shape.page * shape.page
        } else {
            0
        };
        assert_eq!(b.extend_prefix_lens, vec![expected as i32; shape.batch]);
        let mut p = processor(&specs);
        let c = p.commit(&b, None, &result(shape.batch)).unwrap();
        s.advance(&c.events).unwrap();
        assert!(!s.next_execution_plan(true, false).needs_device());
    }
    json!({"commit":commit,"scheduler_finish":finish,"cleanup_plan":cleanup,
        "events":events,"kv_encode":kv_encode,"output_batch":output_batch_time,"encode":encode,
        "metadata_drop":metadata_drop,"wire_drop":wire_drop,
        "before_finish_plan":before_finish_plan,"profile_ms":spans,
        "kv_events":num_events,"kv_event_bytes":kv_event_bytes,"wire_bytes":wire_bytes})
}

#[test]
fn completion_releases_rows_and_preserves_reusable_prefixes() {
    for prefix_cache in [false, true] {
        for steps in [1, 2, 17] {
            for hit_percent in if prefix_cache { vec![0, 50] } else { vec![0] } {
                for events in [false, true] {
                    trial(
                        Shape {
                            batch: 3,
                            prompt: 32,
                            page: 8,
                            steps,
                            prefix_cache,
                            hit_percent,
                            events,
                            profile: false,
                        },
                        true,
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "explicit synchronized completion benchmark; no model or GPU"]
fn benchmark_finish_burst() {
    let config: Value = serde_json::from_str(
        &std::env::var("TS_FINISH_CONFIG").expect("TS_FINISH_CONFIG JSON required"),
    )
    .unwrap();
    let number = |key: &str| {
        config[key]
            .as_u64()
            .unwrap_or_else(|| panic!("{key} required")) as usize
    };
    let flag = |key: &str| {
        config[key]
            .as_bool()
            .unwrap_or_else(|| panic!("{key} required"))
    };
    let shape = Shape {
        batch: number("batch"),
        prompt: number("prompt"),
        steps: number("steps"),
        page: number("page"),
        hit_percent: number("hit_percent"),
        prefix_cache: flag("prefix_cache"),
        events: flag("events"),
        profile: flag("profile"),
    };
    let (warmups, repeats) = (number("warmups"), number("repeats"));
    assert!(
        shape.batch > 0 && shape.prompt >= 2 && shape.steps > 0 && shape.page > 0 && repeats > 0
    );
    assert!(shape.hit_percent < 100 && (shape.prefix_cache || shape.hit_percent == 0));
    assert!(
        shape.batch as u128
            * (shape.prompt as u128 + shape.steps as u128 + 8 * shape.page as u128)
            * 2
            < i32::MAX as u128
    );
    let mut samples = Vec::new();
    for i in 0..warmups + repeats {
        let sample = trial(shape, false);
        if i >= warmups {
            samples.push(sample);
        }
    }
    let output = std::env::var("TS_FINISH_OUTPUT").expect("TS_FINISH_OUTPUT required");
    std::fs::write(
        output,
        json!({"benchmark":"finish_burst","config":config,"samples":samples}).to_string(),
    )
    .unwrap();
}
